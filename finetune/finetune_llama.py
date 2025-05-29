import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- Settings ----
#MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "distilbert/distilgpt2"
TRAIN_PATH = "../../multiwoz-master/parsed/train_sft.jsonl"
VAL_PATH   = "../../multiwoz-master/parsed/test_sft.jsonl"   # Optional: use your own validation/test file!
OUTPUT_DIR = "../../multiwoz-master/parsed/distilgpt-finetuned-multiwoz"
BATCH_SIZE = 1
EPOCHS = 2
MAX_LENGTH = 1024
import transformers
print(transformers.__version__)
print(transformers.__file__)

import sys
print(sys.version)

# ---- 1. Load dataset ----
# # If you want to preprocess, use this block and save to *_sft.jsonl
def preprocess(example):
     dialogue = ""
     for msg in example["messages"]:
         role = msg["role"]
         content = msg["content"].strip().replace("\n", " ")
         if role == "user":
             dialogue += f"<|user|> {content} "
         elif role == "assistant":
             dialogue += f"<|assistant|> {content} "
     return {"text": dialogue.strip()}
# dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})


# from datasets import load_dataset
# data = load_dataset("json", data_files={"all": "all_sft.jsonl"})["all"]
# split = data.train_test_split(test_size=0.1, seed=42)
# split["train"].to_json("train_sft.jsonl")
# split["test"].to_json("val_sft.jsonl")


# ---- 2. Load DATA AND PREPROCESS ----
dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
dataset = dataset.map(preprocess)


# ---- 3. Tokenizer/Model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",         # "cpu" or "auto" (GPU)
    #load_in_4bit=True,         # Remove if you want full precision
)

# ---- 4. LoRA (PEFT) setup ----
"""
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # Adjust as needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
"""
# ---- 5. Tokenization ----
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized = dataset.map(tokenize, batched=True, remove_columns=[col for col in dataset["train"].column_names if col != "text"])

# ---- 6. TrainingArguments & Trainer ----
training_args = TrainingArguments(
    #per_device_train_batch_size=BATCH_SIZE,
    no_cuda=True,
    per_device_train_batch_size=1,      # <--- TRAIN batch size (change as needed)
    per_device_eval_batch_size=1, 
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    save_steps=4220,
    save_total_limit=2,
    eval_strategy="epoch",      # Evaluate every epoch
    save_strategy="epoch",
    eval_steps=None,
    bf16=False,
    fp16=False,                        # Set to False if using CPU
    report_to="none",
    load_best_model_at_end=True,      # Use best checkpoint from eval
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #fp16=False,
    #bf16=False,
)

# ---- 7. Train! ----
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---- 8. Evaluate ----
eval_results = trainer.evaluate()
print("Validation results:", eval_results)
