#!/usr/bin/env python3
"""
Export BlitzKode to GGUF format for Ollama
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pathlib import Path
import json

BLITZKODE = Path("C:/Dev/Projects/BlitzKode")
MODEL_PATH = BLITZKODE / "checkpoints" / "blitzkode-v2" / "checkpoint-4"
OUTPUT_DIR = BLITZKODE / "exported"

print("=" * 60)
print("EXPORTING BLITZKODE TO GGUF")
print("=" * 60)

print("\n[LOADING MODEL]")
base_model_path = "C:/Dev/Projects/BlitzKode/models/qwen1.5b"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
print(f"  Base model: {base_model_path}")

print("  Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print("  Merging adapters...")
model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
model = model.merge_and_unload()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("  Saving merged model...")
model.save_pretrained(str(OUTPUT_DIR / "merged"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "merged"))

print(f"\n[DONE] Merged model saved to: {OUTPUT_DIR / 'merged'}")
print("\nTo convert to GGUF, run:")
print("  git clone --depth 1 https://github.com/ggerganov/llama.cpp")
print("  python llama.cpp/convert.py exported/merged --outfile blitzkode.gguf --outtype q4_k_m")
print("\nThen copy blitzkode.gguf to Ollama models directory")
