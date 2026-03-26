#!/usr/bin/env python3
"""
BlitzKode - Optimized Backend Server
Creator: Sajad
"""

import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import llama_cpp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Paths
BLITZKODE_DIR = Path("C:/Dev/Projects/BlitzKode")
MODEL_PATH = BLITZKODE_DIR / "blitzkode.gguf"
FRONTEND_PATH = BLITZKODE_DIR / "frontend" / "index.html"

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print("=" * 50)
print("BLITZKODE")
print("Creator: Sajad")
print("=" * 50)

# Load model with CPU optimization
print("\nInitializing optimized engine...")
start_time = time.time()
llm = llama_cpp.Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=0,
    n_ctx=2048,
    n_threads=8,
    n_batch=128,
    verbose=False,
    use_mmap=True,
    use_mlock=False,
    seed=-1,
)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f}s (CPU mode)\n")

# Thread pool for generation
executor = ThreadPoolExecutor(max_workers=1)

# FastAPI app
app = FastAPI(title="BlitzKode API", version="1.6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request validation
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.5
    max_tokens: int = 256
    top_p: float = 0.95
    top_k: int = 20
    repeat_penalty: float = 1.05

MAX_PROMPT_LENGTH = 4000

# System prompt
SYSTEM_PROMPT = """<|im_start|>system
You are BlitzKode, an AI coding assistant created by Sajad. You are an expert in Python, JavaScript, Java, C++, and all programming languages. Write clean, efficient, and well-documented code. Keep responses concise and practical.<|im_end|>"""

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_PATH))

@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy", "model_loaded": True, "version": "1.6"})

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    if len(req.prompt) > MAX_PROMPT_LENGTH:
        return JSONResponse({"error": f"Prompt too long. Max {MAX_PROMPT_LENGTH} chars."}, status_code=400)
    
    full_prompt = f"{SYSTEM_PROMPT}<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        result = await loop.run_in_executor(executor, generate_sync, full_prompt, req)
    else:
        result = generate_sync(full_prompt, req)
    
    return result

def generate_sync(full_prompt, req):
    try:
        result = llm(
            full_prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<|im_end|>", "<|im_start|>user"],
        )
        response = result["choices"][0]["text"].strip()
        return JSONResponse({
            "response": response,
            "creator": "Sajad",
            "model": "BlitzKode",
            "version": "1.6"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    if len(req.prompt) > MAX_PROMPT_LENGTH:
        return JSONResponse({"error": f"Prompt too long. Max {MAX_PROMPT_LENGTH} chars."}, status_code=400)
    
    full_prompt = f"{SYSTEM_PROMPT}<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate_tokens():
        try:
            for token in llm(
                full_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repeat_penalty=req.repeat_penalty,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["<|im_end|>", "<|im_start|>user"],
                stream=True,
            ):
                if token.get("choices"):
                    text = token["choices"][0].get("text", "")
                    if text:
                        yield f"data: {repr(text)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {repr(str(e))}\n\n"
    
    return StreamingResponse(generate_tokens(), media_type="text/event-stream")

@app.get("/info")
async def info():
    return JSONResponse({
        "name": "BlitzKode",
        "creator": "Sajad",
        "version": "1.6",
        "status": "ready",
        "mode": "CPU (optimized)",
        "endpoints": {
            "generate": "POST /generate",
            "stream": "POST /generate/stream",
            "health": "GET /health",
            "info": "GET /info"
        }
    })

if __name__ == "__main__":
    import asyncio
    print("Open: http://localhost:7860")
    print("API: http://localhost:7860/info\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
