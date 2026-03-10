#!/usr/bin/env python3
"""
Professional Web Interface for BlitzKode - Gradio 6 Compatible
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import gradio as gr
import sys

MODEL_PATH = "C:/Dev/Projects/BlitzKode/exported/merged"

print("Loading BlitzKode Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print("BlitzKode Ready!")
sys.stdout.flush()

def generate(prompt, temperature, max_tokens):
    full_prompt = f"<|im_start|>system\nYou are BlitzKode, an expert coding assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            temperature=temperature,
            max_new_tokens=int(max_tokens),
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(full_prompt):].strip()
    return response

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Write a Python function..."),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature"),
        gr.Slider(minimum=128, maximum=2048, value=1024, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Response", lines=20),
    title="BlitzKode - AI Coding Assistant",
    description="Your expert coding assistant powered by fine-tuned Qwen2.5-1.5B",
)

demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
