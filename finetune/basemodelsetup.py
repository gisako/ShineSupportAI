# base model setup

import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Change as needed
#MINI_MODEL_NAME = "meta-llama/Llama-3.2-1B"

MINI_MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
DATA_PATH = "train_data.jsonl"             # Update to your dataset


def login_hf():
    print("Logging in to Hugging Face CLI...")
    subprocess.run(["huggingface-cli", "login"])

def download_model():
    print(f"Downloading model/tokenizer: {MODEL_NAME} (Standard, uses available GPU/CPU RAM)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer downloaded and cached locally.")

def local_download_disk_offload():
    print(f"Downloading model/tokenizer with disk offload: {MODEL_NAME}")
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    except ImportError:
        print("Please install the accelerate package with: pip install accelerate")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    print("Initializing empty model (for disk offload)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    print("Loading model weights with disk offload. This may take some time...")
    offload_dir = "offload_dir"
    model = load_checkpoint_and_dispatch(
        model,
        MODEL_NAME,
        device_map="auto",
        offload_folder=offload_dir,
        offload_state_dict=True
    )
    print(f"Model downloaded with disk offload to '{offload_dir}'. Tokenizer cached as well.")

def download_mini_model():
 	print(f"Downloading model/tokenizer: {MINI_MODEL_NAME} (Standard, uses available GPU/CPU RAM)")
 	model = AutoModelForCausalLM.from_pretrained(MINI_MODEL_NAME,device_map="cpu")
 	tokenizer = AutoTokenizer.from_pretrained(MINI_MODEL_NAME)
 	print("Model and tokenizer downloaded and cached locally.")


def prepare_data():
    print(f"Preparing dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files={"train": DATA_PATH, "validation": DATA_PATH})
    print(dataset)
    print("Sample:", dataset["train"][0])

def tokenize_data():
    print("Tokenizing dataset...")
    dataset = load_dataset("json", data_files={"train": DATA_PATH, "validation": DATA_PATH})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(example):
        # Assumes 'messages' field (adjust as needed)
        return tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk("tokenized_dataset")
    print("Tokenized data saved to disk as 'tokenized_dataset'.")

def fine_tune():
    print("Starting fine-tuning...")
    tokenized_dataset = load_dataset("json", data_files={"train": DATA_PATH, "validation": DATA_PATH})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./finetuned-llama3",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_steps=100,
        save_total_limit=2
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("finetuned-llama3-support-bot")
    print("Fine-tuned model saved as 'finetuned-llama3-support-bot'.")

def test_model():
    print("Testing model inference...")
    model = AutoModelForCausalLM.from_pretrained("finetuned-llama3-support-bot", torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_text = input("Enter a prompt for the bot: ")
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    print("\nBot response:\n", tokenizer.decode(output[0], skip_special_tokens=True))

def main():
    while True:
        print("\n==== Llama 3 Hugging Face CLI ====")
        print("1. Login to Hugging Face")
        print("2. Download Model & Tokenizer (Standard)")
        print("3. Download Model with Disk Offload (for low GPU RAM)")
        print("4. Prepare Data")
        print("5. Tokenize Data")
        print("6. Fine-Tune Model")
        print("7. Test Model Inference")
        print("8. Downlad Mini Model and Tokenizer")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            login_hf()
        elif choice == "2":
            download_model()
        elif choice == "3":
            local_download_disk_offload()
        elif choice == "4":
            prepare_data()
        elif choice == "5":
            tokenize_data()
        elif choice == "6":
            fine_tune()
        elif choice == "7":
            test_model()
        elif choice == "8":
            download_mini_model()
        elif choice == "0":
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
