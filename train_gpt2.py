import os
import json
import torch
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# -----------------
# Device setup
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# -----------------
# Model path
# -----------------
model_path = "./gpt2_chatbot"

# -----------------
# Load tokenizer & model
# -----------------
if os.path.exists(model_path) and os.path.isdir(model_path):
    print("🔄 Loading existing model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
else:
    print("🚀 Starting fresh from base GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    os.makedirs(model_path, exist_ok=True)

tokenizer.pad_token = tokenizer.eos_token
model.to(device)

# -----------------
# Apply LoRA
# -----------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"]
)
model = get_peft_model(model, lora_config)
print("✅ LoRA applied. Only low-rank weights will be trained.")

# -----------------
# Load profile facts
# -----------------
with open("profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

profile_text = "<|profile|>\n"
for key, value in profile.items():
    if isinstance(value, list):
        value_str = ", ".join(map(str, value))
    else:
        value_str = str(value)
    profile_text += f"- {key}: {value_str}\n"
profile_text += "<|endprofile|>\n\n"

# -----------------
# Load dialogues
# -----------------
with open("chat_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def format_conversation(conv):
    # Include profile at the start for grounding
    return profile_text + "\n".join(conv["dialogue"]) + tokenizer.eos_token

dataset = [{"text": format_conversation(conv)} for conv in data]
dataset = Dataset.from_list(dataset).shuffle(seed=42)

# Train/validation split
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# -----------------
# Tokenization
# -----------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_attention_mask=True,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# -----------------
# Data collator
# -----------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------
# Training setup
# -----------------
training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=False,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=100,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    logging_dir="./logs",
    eval_strategy="epoch",  # correct argument
    report_to="none",
    prediction_loss_only=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# -----------------
# Training
# -----------------
print("🚀 Starting LoRA training...")
trainer.train()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("✅ Training complete. Model saved!")
