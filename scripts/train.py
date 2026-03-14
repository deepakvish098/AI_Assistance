import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

#model_name = "distilgpt2"  // it It just learns to continue text in a similar style to what it was trained on

model_name="google/flan-t5-small"
# Fix 1: Added trust_remote_code=True (required for phi-2)
# Fix 2: Set pad_token = eos_token (phi-2 has no pad token by default)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, config)

# Fix 3: Use absolute path based on script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = load_dataset("json", data_files=os.path.join(BASE_DIR, "data", "dataset.json"))

# Fix 4: Tokenize dataset — converts question/answer into input_ids + labels
def tokenize(example):
    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, remove_columns=["question", "answer"])

# Fix 5: Use absolute output_dir + reduced batch size to avoid OOM
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(BASE_DIR, "models"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=5,
    save_strategy="no",
    dataloader_pin_memory=False,
    use_cpu=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()

# Fix 6: Use absolute paths for saving
model.save_pretrained(os.path.join(BASE_DIR, "models", "lora_model"))
tokenizer.save_pretrained(os.path.join(BASE_DIR, "models", "lora_model"))
