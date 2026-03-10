#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import gradio as gr

MODEL_PATH = "C:/Dev/Projects/BlitzKode/checkpoints/dpo-v1/final"

print("Loading BlitzKode model...")

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

print("BlitzKode Ready! Starting web interface...")

def generate_response(prompt, temperature=0.7, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response

demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Write a Python function to..."),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature"),
        gr.Slider(minimum=64, maximum=1024, value=512, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Response", lines=20),
    title="BlitzKode - Coding Assistant",
    description="Your fine-tuned coding assistant based on Qwen2.5-1.5B",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
