from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,  
    dtype=None,           
    load_in_4bit=True     
)
dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]")  # First 1000 articles

training_args = TrainingArguments(
    output_dir="./results",           
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()