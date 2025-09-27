"""
train_phase2.py — Continue fine-tuning (Phase 2)
- Loads merged Phase1 model
- Trains on Constitution + IPC + Laws dataset
- Saves new LoRA checkpoint
"""

import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# ✅ Config
cfg = {
    "model_name_or_path": "outputs/merged_phase1",  # merged Phase 1 model
    "output_dir": "outputs_phase2",
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 5, # updated by 3 to 5
    "learning_rate": 5e-5,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "max_seq_length": 512,
    "use_4bit": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05
}

def preprocess(dataset, tokenizer, max_length=512):
    def tok_fn(ex):
        prompt = ""
        for msg in ex["messages"]:
            role = msg["role"]
            content = msg["content"]
            prompt += f"[{role.capitalize()}]: {content}\n"

        tok = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        labels = tok["input_ids"].copy()
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels
        }
    return dataset.map(tok_fn, batched=False, remove_columns=["messages"])

def main():
    model_name = cfg["model_name_or_path"]

    # Load Phase 2 dataset
    ds = load_from_disk("data/phase2_chat")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    train_ds = preprocess(ds, tokenizer, max_length=cfg["max_seq_length"])

    # Load merged Phase 1 model
    load_kwargs = {"device_map": "auto"}
    if cfg.get("use_4bit", False):
        load_kwargs.update({"load_in_4bit": True, "bnb_4bit_use_double_quant": True})

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **load_kwargs)
    model.resize_token_embeddings(len(tokenizer))

    # Setup new LoRA for Phase 2
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=cfg.get("lora_dropout", 0.05)
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=False,
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(os.path.join(cfg["output_dir"], "final_lora_phase2"))
    print("✅ Phase 2 Training complete. Final model saved.")

if __name__ == "__main__":
    main()
