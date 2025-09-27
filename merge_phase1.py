# merge_phase1.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base + Phase 1 LoRA path
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
lora_path = "outputs/final_lora"

# Load base + LoRA
print("🔄 Loading base + Phase 1 LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, lora_path)

# Merge weights into base model
print("🔄 Merging LoRA into base model...")
merged_model = model.merge_and_unload()

# Save merged model
save_path = "outputs/merged_phase1"
merged_model.save_pretrained(save_path)
AutoTokenizer.from_pretrained(base_model_name).save_pretrained(save_path)

print(f"✅ Merged model saved at: {save_path}")
