# merge_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Paths
base_model = "outputs/merged_phase1"            # after Phase 1
lora_model = "outputs_phase2/final_lora_phase2" # Phase 2 LoRA adapter
merged_model_path = "outputs/final_merged_model" # final single model

print("🔄 Loading base + LoRA...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=False,  # must disable quantization when merging
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_model)

print("🔗 Merging weights...")
merged_model = model.merge_and_unload()

print("💾 Saving merged model...")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"✅ Done! Final model saved at {merged_model_path}")
