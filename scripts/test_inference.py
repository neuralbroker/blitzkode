#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_PATH = "C:/Dev/Projects/BlitzKode/checkpoints/dpo-v1/final"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading base model...")
peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("\n" + "="*60)
print("Testing model...")
print("="*60)

prompt = "Write a Python function to find the two sum of indices that add up to target."

print(f"\nPrompt: {prompt}\n")
print("Response:")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
print("\n" + "="*60)
print("Test complete!")
