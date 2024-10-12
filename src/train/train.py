from huggingface_hub import HfApi
from transformers import TrainingArguments, Trainer
import uuid
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import random
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import concatenate_datasets
from transformers import BitsAndBytesConfig


class DataCollatorForImageCaptioning:
    def __init__(self, processor, image_column_name="image", text_column_name="text"):
        self.processor = processor
        self.image_column_name = image_column_name
        self.text_column_name = text_column_name

    def __call__(self, dataset):
        texts = []
        images = []
        assistant_responses = []  # To track assistant responses for proper masking
        for data in dataset:
            image = data[self.image_column_name]
            questions = [
                "What is this image about?",
                "Can you describe this image?",
                "What is in this image?",
                "Identify the food/ingredients in this image." "",
            ]
            question = questions[random.randint(0, len(questions) - 1)]
            answer = data[self.text_column_name]

            messages = [
                # {
                #     "role": "system",
                #     "content": [{"type": "text", "text": system_message}],
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            # Convert messages to the desired text format using the chat template
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False
            )
            texts.append(text.strip())
            images.append([image])
            assistant_responses.append(answer)  # Track assistant responses

        # Tokenize and process batch
        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # Prepare labels: we will mask non-assistant tokens for the model to predict
        labels = batch["input_ids"].clone()

        # For each example, find assistant tokens and mask everything else
        for i, (input_ids, assistant_response) in enumerate(
            zip(batch["input_ids"], assistant_responses)
        ):
            # Tokenize just the assistant response
            assistant_tokens = self.processor.tokenizer(
                assistant_response, return_tensors="pt"
            )["input_ids"].squeeze()

            # Find where the assistant tokens start in the input sequence
            # This method ensures we match the tokenized assistant response, not the input text
            start_idx = self.find_subsequence(input_ids, assistant_tokens)

            if start_idx is not None:
                # Mask everything except the assistant tokens: Help with training efficiency (cross-entropy loss)
                labels[
                    i, :start_idx
                ] = -100  # Ignore everything before the assistant tokens
                labels[
                    i, start_idx + len(assistant_tokens) :
                ] = -100  # Ignore everything after the assistant tokens

        # Assign masked labels to the batch
        batch["labels"] = labels
        return batch

    def find_subsequence(self, sequence, subsequence):
        """Find the start index of a subsequence (assistant response) in a sequence (input_ids)
        Usefull to mask the input_ids for the model to predict only the assistant response.
        """
        seq_len = len(sequence)
        sub_len = len(subsequence)

        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i : i + sub_len], subsequence):
                return i
        return None


def train():
    dataset1 = load_dataset(
        "blanchon/food_500_llm_subset",
        split="train",
    ).shuffle()
    dataset2 = load_dataset(
        "eBoreal/food-500-enriched",
        split="train",
    ).shuffle()

    dataset = concatenate_datasets([dataset1, dataset2])  # type: ignore

    # train_dataset is 90% of the dataset
    train_dataset = dataset.select(range(int(len(dataset) * 0.9)))
    # eval_dataset is 10% of the dataset
    eval_dataset = dataset.select(range(int(len(dataset) * 0.1)))

    # Hugging Face model id
    model_id = "mistral-community/pixtral-12b"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    CHAT_TEMPLATE = """
    {%- for message in messages %}
        {%- if message.role == "user" %}
            <s>[INST]
            {%- for item in message.content %}
                {%- if item.type == "text" %}
                    {{ item.text }}
                {%- elif item.type == "image" %}
                    \n[IMG]
                {%- endif %}
            {%- endfor %}
            [/INST]
        {%- elif message.role == "assistant" %}
            {%- for item in message.content %}
                {%- if item.type == "text" %}
                    {{ item.text }}
                {%- endif %}
            {%- endfor %}
            </s>
        {%- endif %}
    {%- endfor %}
    """

    # Set the chat template for the tokenizer
    processor.chat_template = CHAT_TEMPLATE.replace("    ", "")
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    data_collator = DataCollatorForImageCaptioning(
        processor, image_column_name="image", text_column_name="text"
    )

    lora_config = LoraConfig(
        r=32,  # Rank (usually 8, 16, or 32 depending on the model size and task)
        lora_alpha=32,  # Scaling factor for the low-rank updates
        use_rslora=True,  # Use rank stabilizing LoRA, adjust lr depending on the rank
        target_modules="all-linear",
        # modules_to_save=["lm_head"], # Fully train the linear head and embedding layer for images
        lora_dropout=0.1,  # Dropout for low-rank adapter layers
        bias="none",  # Bias in adapter layers: "none", "all", "lora_only"
        task_type="CAUSAL_LM",  # Task type: "CAUSAL_LM", "SEQ_2_SEQ", ...
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    epochs = 2
    lr = 3e-5
    schedule = "constant"
    random_uuid = uuid.uuid4().hex[:4]
    run_name = f"pixtral-nutrition-{lr}_r-{epochs}_epochs-{schedule}_{random_uuid}"

    training_args = TrainingArguments(
        max_steps=100,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Increase if you have more VRAM
        per_device_eval_batch_size=4,  # Increase if you have more VRAM
        gradient_accumulation_steps=1,  # Decrease if you have more VRAM
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        output_dir="model",
        evaluation_strategy="steps",
        eval_steps=10,
        lr_scheduler_type=schedule,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=4,
        bf16=True,
        hub_model_id=f"blanchon/{run_name}",
        push_to_hub=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name=run_name,
        logging_dir=f"./logs/{run_name}",
        gradient_checkpointing=True,  # should reduce VRAM requirements
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    merged_model = model.merge_and_unload()
    merged_model.push_to_hub(f"blanchon/{run_name}", use_temp_dir=True)

    api = HfApi()

    api.upload_folder(
        folder_path="./logs/",
        path_in_repo="logs",
        repo_id=f"blanchon/{run_name}",
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="./preprocessor_config.json",
        path_in_repo="preprocessor_config.json",
        repo_id=f"blanchon/{run_name}",
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="./tokenizer.json",
        path_in_repo="tokenizer.json",
        repo_id=f"blanchon/{run_name}",
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="./tokenizer_config.json",
        path_in_repo="tokenizer_config.json",
        repo_id=f"blanchon/{run_name}",
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="./special_tokens_map.json",
        path_in_repo="special_tokens_map.json",
        repo_id=f"blanchon/{run_name}",
        repo_type="model",
    )


if __name__ == "__main__":
    train()
