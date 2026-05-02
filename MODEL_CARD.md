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

**BlitzKode** is a fine-tuned AI coding assistant built by **Sajad** using the Qwen2.5-1.5B base model. It's packaged as a GGUF format model for fast local inference with llama.cpp.

> Created by [Abdulla Sajad](https://github.com/neuralbroker)  
> Project: [neuralbroker/blitzkode](https://github.com/neuralbroker/blitzkode)

---

## Model Summary

| Property | Value |
|----------|-------|
| **Model Name** | BlitzKode |
| **Version** | 2.0 |
| **Base Model** | Qwen/Qwen2.5-1.5B-Instruct |
| **Model Format** | GGUF (F16, ~3GB) |
| **Primary Runtime** | llama.cpp / llama-cpp-python |
| **Artifact** | `blitzkode.gguf` |
| **Context Window** | 2048 tokens |
| **Creator** | Sajad |
| **License** | MIT (also see Qwen2.5 upstream license) |

---

## Architecture

- **Model Type**: Transformer-based LLM (1.5B parameters)
- **Architecture**: Qwen2
- **Quantization**: GGUF F16 (~3GB)
- **Vocabulary**: 151,936 tokens
- **Inference**: CPU/GPU with llama.cpp (configurable via BLITZKODE_GPU_LAYERS)

---

## Training Pipeline

BlitzKode was fine-tuned through a 4-stage pipeline:

### 1. SFT (Supervised Fine-Tuning)
Applies LoRA fine-tuning to coding-style prompts and responses using PEFT library.

### 2. Reward-based SFT continuation
Applies additional SFT with heuristic reward functions for code correctness, formatting, and reasoning. Note: this stage uses standard SFT training, not a full GRPO implementation.

### 3. DPO (Direct Preference Optimization)
Trains on handcrafted preference pairs to improve clarity and answer quality.

### 4. Merge & Export
Merges LoRA adapters into base model and converts to GGUF format.

### Training Frameworks
- HuggingFace Transformers
- PEFT (LoRA)
- TRL (DPO/GRPO)
- llama.cpp (inference/export)

---

## Training Data

Custom curated coding datasets covering:
- Algorithm implementation
- Data structures
- Code explanations
- Programming concepts
- Bug fixing scenarios

---

## Features

- **Multi-language Code Generation** - Python, JavaScript, Java, C++, TypeScript, SQL
- **Code Explanation** - Clear comments and documentation
- **Bug Fixing** - Debug and fix code issues
- **Algorithm Assistance** - Data structures and algorithms
- **Offline Operation** - Runs locally without internet
- **Fast Inference** - Optimized CPU inference
- **Modern UI** - Professional dark interface

---

## Intended Use

### Best For
- Local offline coding assistance
- Algorithm and data structure help
- Code generation and explanation
- Educational programming support
- Code review and debugging

### Out of Scope
- Production code without expert review
- Security-critical applications
- Multi-modal tasks (images not supported)
- Long-context repository analysis

---

## Usage

See the [project README](https://github.com/neuralbroker/blitzkode) for full setup instructions, API reference, Docker deployment, and environment variable configuration.

### Quick Start

```bash
pip install llama-cpp-python fastapi uvicorn pydantic
python server.py
# Open http://localhost:7860
```

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

## Limitations

- **Text-only input** - No image/vision support
- **2048 token context** - CPU-friendly but limited
- **Verify outputs** - Always review generated code before use
- **Small model** - May occasionally produce incorrect code

---

## Project Structure

```
BlitzKode/
├── server.py              # FastAPI backend
├── blitzkode.gguf         # Quantized model (~3GB)
├── frontend/index.html    # Web UI
├── tests/test_server.py  # HTTP tests
├── scripts/               # Training + dataset scripts
│   ├── train_sft.py       # SFT training
│   ├── train_reward_sft.py # Reward-based SFT continuation
│   ├── train_dpo.py       # DPO training
│   ├── export_gguf.py     # Model export
│   └── test_inference.py  # Inference test
├── datasets/              # Training data
└── README.md              # Full project docs
```

Full project documentation: [README.md](https://github.com/neuralbroker/blitzkode)

---

## License

MIT License - See README.md for details.

Also comply with upstream Qwen base model license when redistributing.

---

## Contact

- **GitHub**: https://github.com/neuralbroker/blitzkode
- **Portfolio**: https://neuralbroker.vercel.app
- Issues and contributions welcome!

---

## Citation

```bibtex
@software{blitzkode2026,
  author = {Sajad},
  title = {BlitzKode - AI Coding Assistant},
  year = {2026},
  url = {https://github.com/neuralbroker/blitzkode}
}
```
