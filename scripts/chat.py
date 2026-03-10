#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys

MODEL_PATH = "C:/Dev/Projects/BlitzKode/checkpoints/dpo-v1/final"

print("Loading BlitzKode...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("BlitzKode Ready!")
print("=" * 50)
print("Type 'quit' or 'exit' to stop\n")

while True:
    try:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        if not prompt.strip():
            continue
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        print(f"\nBlitzKode: {response}\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
