---
language:
- en
library_name: llama-cpp-python
pipeline_tag: text-generation
tags:
- code-generation
- coding-assistant
- gguf
- llama.cpp
- qwen2.5
- python
- javascript
- fine-tuned
base_model:
- Qwen/Qwen2.5-1.5B-Instruct
---

# BlitzKode

<p align="center">
  <img src="Screenshot 2026-03-26 122611.png" alt="BlitzKode UI" width="800"/>
</p>

**BlitzKode** is a locally fine-tuned AI coding assistant built by **Sajad** using the Qwen2.5-1.5B base model. It's packaged as a GGUF format model for fast local inference with llama.cpp.

> Created by [Abdulla Sajad](https://github.com/sajadkoder)  
> Project: [sajadkoder/blitzkode](https://github.com/sajadkoder/blitzkode)

---

## Model Summary

| Property | Value |
|----------|-------|
| **Model Name** | BlitzKode |
| **Version** | 1.6 (CPU optimized) |
| **Base Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Model Format** | GGUF (F16, ~3GB) |
| **Primary Runtime** | llama.cpp / llama-cpp-python |
| **Artifact** | `blitzkode.gguf` |
| **Context Window** | 2048 tokens |
| **Creator** | Sajad |
| **License** | MIT |

---

## Architecture

- **Model Type**: Transformer-based LLM (1.5B parameters)
- **Architecture**: Qwen2
- **Quantization**: GGUF F16 (~3GB)
- **Vocabulary**: 151,936 tokens
- **Inference**: CPU-optimized with llama.cpp

---

## Training Pipeline

BlitzKode was fine-tuned through a 4-stage pipeline:

### 1. SFT (Supervised Fine-Tuning)
- **Script**: `scripts/train_sft.py`
- Applies LoRA fine-tuning to coding-style prompts and responses
- Uses PEFT library for efficient parameter-efficient training

### 2. GRPO (Group Relative Policy Optimization)
- **Script**: `scripts/train_grpo.py`
- Uses heuristic reward functions:
  - `correctness_reward` - Code correctness
  - `format_reward` - Proper code formatting
  - `reasoning_reward` - Logic and reasoning

### 3. DPO (Direct Preference Optimization)
- **Script**: `scripts/train_dpo.py`
- Trains on handcrafted chosen/rejected preference pairs
- Improves clarity and answer quality

### 4. Merge & Export
- **Script**: `scripts/export_gguf.py`
- Merges LoRA adapters into base model
- Converts to GGUF format for fast inference

### Training Frameworks
- HuggingFace Transformers
- PEFT (LoRA)
- TRL (DPO/GRPO)
- llama.cpp (inference/export)

---

## Training Data

### Local Datasets
- `datasets/raw/blitzkode_sft_v1.json` - Seed samples
- `datasets/raw/blitzkode_sft_full.json` - Extended coding samples

### Data Categories
- Arrays and hash maps
- Linked lists
- Trees and graph traversal
- Dynamic programming
- Sorting and searching
- Stack and queue implementations
- Interview-style coding problems
- Code explanations

### Optional External Sources
The project can optionally incorporate:
- CodeAlpaca-20k
- GSM8K
- MetaMathQA
- MathInstruct

---

## Features

- **Multi-language Code Generation** - Python, JavaScript, Java, C++, TypeScript, HTML/CSS, SQL
- **Code Explanation** - Clear comments and documentation
- **Bug Fixing** - Debug and fix code issues
- **Algorithm Help** - Data structures and algorithms
- **Offline Operation** - Runs locally without internet
- **Fast Inference** - Optimized CPU inference
- **Modern UI** - ChatGPT-style dark interface

---

## Intended Use

### Best For
- Local offline coding assistance
- Algorithm and data structure help
- Code generation and explanation
- Educational programming support
- Lightweight code review
- Bug detection and fixing

### Out of Scope
- Production code without expert review
- Security-critical applications
- Multi-modal tasks (images not supported)
- Long-context repository analysis
- Real-time high-assurance systems

---

## API & Usage

### Running the Server

```bash
# Install dependencies
pip install llama-cpp-python fastapi uvicorn pydantic

# Start server
python server.py

# Open browser
# http://localhost:7860
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/info` | GET | API info |
| `/generate` | POST | Generate response |
| `/generate/stream` | POST | Stream tokens |

### API Example

```bash
# Generate code
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in python"}'

# Stream response
curl -X POST http://localhost:7860/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function"}'
```

### Python Usage

```python
from llama_cpp import Llama

llm = Llama(
    model_path="blitzkode.gguf",
    n_ctx=2048,
    n_threads=8,
)

prompt = """<|im_start|>system
You are BlitzKode, a coding assistant.<|im_end|>
<|im_start|>user
Write a hello world in Python<|im_end|>
<|im_start|>assistant
"""

result = llm(prompt, max_tokens=256)
print(result["choices"][0]["text"])
```

---

## Prompt Format

Uses ChatML-style template:

```
<|im_start|>system
You are BlitzKode, an AI coding assistant created by Sajad. You are an expert in Python, JavaScript, Java, C++, and other programming languages. Write clean, efficient, and well-documented code. Keep responses concise and practical.<|im_end|>
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
```

---

## Configuration

The server supports environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BLITZKODE_MODEL_PATH` | `blitzkode.gguf` | Model file path |
| `BLITZKODE_FRONTEND_PATH` | `frontend/index.html` | UI path |
| `BLITZKODE_HOST` | `0.0.0.0` | Server host |
| `BLITZKODE_PORT` | `7860` | Server port |
| `BLITZKODE_THREADS` | CPU count | CPU threads |
| `BLITZKODE_N_CTX` | `2048` | Context window |
| `BLITZKODE_BATCH` | `128` | Batch size |
| `BLITZKODE_MAX_PROMPT_LENGTH` | `4000` | Max prompt chars |

---

## Limitations

- **Text-only input** - No image/vision support
- **2048 token context** - CPU-friendly but limited
- **Small model** - May produce incorrect code occasionally
- **No formal benchmarks** - Not evaluated on standard datasets
- **Quantization loss** - F16 quantization may reduce accuracy
- **Verify outputs** - Always review generated code before use

---

## Project Structure

```
BlitzKode/
├── server.py              # FastAPI backend (v1.6)
├── blitzkode.gguf         # Quantized model (~3GB)
├── frontend/
│   └── index.html        # Web UI
├── tests/
│   └── test_server.py    # HTTP tests
├── scripts/
│   ├── train_sft.py       # SFT training
│   ├── train_grpo.py     # GRPO training
│   ├── train_dpo.py      # DPO training
│   ├── export_gguf.py    # Model export
│   └── test_inference.py # Inference test
├── checkpoints/          # LoRA checkpoints
├── datasets/             # Training data
├── MODEL_CARD.md         # This file
└── README.md             # Project docs
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.6 | Current | CPU optimization, faster inference |
| 1.5 | Earlier | Added streaming support |
| 1.0 | Initial | Base model release |

---

## License

MIT License - See README.md for details.

Also comply with upstream Qwen base model license when redistributing.

---

## Contact

- **GitHub**: https://github.com/sajadkoder/blitzkode
- **Portfolio**: https://sajadkoder.vercel.app
- Issues and contributions welcome!

---

## Citation

```bibtex
@software{blitzkode2026,
  author = {Sajad},
  title = {BlitzKode - AI Coding Assistant},
  year = {2026},
  url = {https://github.com/sajadkoder/blitzkode}
}
```
