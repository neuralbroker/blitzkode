---
language:
- en
library_name: llama-cpp-python
pipeline_tag: text-generation
tags:
- code
- coding-assistant
- gguf
- llama.cpp
- qwen2.5
- python
- javascript
base_model:
- Qwen/Qwen2.5-1.5B-Instruct
---

# BlitzKode

**BlitzKode** is a locally fine-tuned coding assistant packaged as a GGUF model for fast local inference with `llama.cpp` and `llama-cpp-python`.

This model card is written conservatively and reflects what can be supported by the repository contents and checkpoint metadata currently included with the project.

> Developed by [Abdulla Sajad](https://github.com/sajadkoder)  
> Project repository: [sajadkoder/blitzkode](https://github.com/sajadkoder/blitzkode)  
> Portfolio: [sajadkoder.vercel.app](https://sajadkoder.vercel.app)

## Model Summary

| Property | Value |
| --- | --- |
| Model name | BlitzKode |
| Version | 1.6 (CPU optimized) |
| Base model family | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model format | GGUF (F16, ~3GB) |
| Primary runtime | `llama.cpp` / `llama-cpp-python` |
| Served artifact | `blitzkode.gguf` |
| Default context | 2048 tokens |
| Intended language | English (code generation) |

## Important Provenance Note

The repository includes a clear staged fine-tuning workflow and several checkpoint families, but it does **not** include a release manifest that pins the checked-in `blitzkode.gguf` artifact to one exact training checkpoint and dataset snapshot.

What can be verified:

- The shipped server serves `blitzkode.gguf` by default.
- The available LoRA checkpoints in the repo point to a local base model directory at `models/qwen1.5b`.
- Training scripts consistently describe that base model as **Qwen 2.5 1.5B Instruct**.
- The repository contains SFT, GRPO-style, DPO, and GGUF export scripts.

What should **not** be claimed without additional release metadata:

- A fully reproducible checkpoint-to-GGUF lineage
- Benchmark or pass-rate numbers
- Formal safety evaluation results
- Exact dataset composition for the released GGUF
- The `Qwen2.5-Coder-1.5B-Instruct` base variant specifically

## Model Details

### Base Model

Checkpoint metadata and training scripts point to the **Qwen 2.5 1.5B Instruct** family, not the coder-specific variant.

Relevant repository signals:

- `scripts/train_v2.py` references `Qwen/Qwen2.5-1.5B-Instruct`
- `scripts/train_sft.py`, `scripts/train_grpo.py`, and `scripts/train_dpo.py` all describe the model as Qwen 2.5 1.5B Instruct
- `adapter_config.json` files in the saved checkpoints point to a local base-model path: `C:/Dev/Projects/BlitzKode/models/qwen1.5b`

### Runtime Format

The repository currently supports two artifact forms:

- **LoRA checkpoints** under `checkpoints/`
- **Quantized GGUF** for fast local inference via `llama.cpp`

The default app path in `server.py` loads `blitzkode.gguf` through `llama_cpp.Llama`.

## Training Pipeline

The repository contains a staged fine-tuning workflow:

1. **SFT (Supervised Fine-Tuning)**  
   `scripts/train_sft.py` applies LoRA fine-tuning to locally curated coding-style prompts and responses.

2. **GRPO-style continuation**  
   `scripts/train_grpo.py` applies an additional training stage using heuristic reward functions named `correctness_reward`, `format_reward`, and `reasoning_reward`.

3. **DPO (Direct Preference Optimization)**  
   `scripts/train_dpo.py` trains on handcrafted chosen/rejected preference pairs for clearer and more preferred answers.

4. **Merge and GGUF export**  
   `scripts/export_gguf.py` merges adapters into the base model and prepares the model for GGUF conversion through `llama.cpp`.

### Training Frameworks

The project uses:

- Hugging Face `transformers`
- `peft` for LoRA adapters
- `trl` for DPO and GRPO-style experimentation
- `llama.cpp` for local inference/export workflow

## Training Data

The dataset signals in the repository are strongest for **algorithmic coding tasks**, **data structures**, and **code explanation with short complexity notes**.

Local data artifacts currently present in the repo include:

- `datasets/raw/blitzkode_sft_v1.json` with 3 seed samples
- `datasets/raw/blitzkode_sft_full.json` with 24 local coding samples

Observed prompt types in local datasets include:

- arrays and hash maps
- linked lists
- trees and graph traversal
- dynamic programming
- sorting and searching
- basic data structures such as stacks and queues
- common interview-style coding problems

The repository also contains dataset-builder scripts that can optionally incorporate external sources such as:

- `sahil2801/CodeAlpaca-20k`
- `openai/gsm8k`
- `meta-math/MetaMathQA`
- `meta-math/MathInstruct`

### Data Quality Note

The project should **not** currently be described as using a fully documented curated dataset where every sample includes reasoning traces and test cases. The checked-in local datasets are much smaller and more lightweight than that description suggests, and the optional external sources are not pinned to a final release manifest.

## Intended Use

BlitzKode is best suited for:

- local offline coding assistance
- algorithm and data-structure help
- concise code generation and explanation
- educational experimentation with local model serving
- lightweight code review and bug-fix suggestions

## Out-of-Scope Use

BlitzKode is **not** intended for:

- unsupervised production code generation
- security-critical or safety-critical code without expert review
- long-context repository reasoning across large codebases
- claims of benchmarked coding performance without separate evaluation evidence
- domains requiring real-time or high-assurance factual accuracy

## Limitations

- The default server configuration uses a 2048-token context window for CPU-friendly inference.
- The repository does not include a formal benchmark suite for the model itself.
- The GRPO-style stage uses heuristic reward functions; it does not execute generated code against a comprehensive correctness harness.
- Small-model and quantization constraints can reduce factual reliability, long-horizon reasoning, and code accuracy.
- The project snapshot includes multiple experimental checkpoint families, which makes release provenance less precise than a typical production model release.

## Evaluation

No formal model-quality evaluation report is included in the top-level repository.

Operational verification currently available in the repo includes:

- HTTP smoke tests for the serving layer in `tests/test_server.py`
- a local checkpoint inference smoke script in `scripts/test_inference.py`

These checks validate runtime behavior, not coding benchmark performance.

## Usage

### Python (`llama-cpp-python`)

```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_path = hf_hub_download(
    repo_id="sajadkoder/blitzkode",
    filename="blitzkode.gguf",
)

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    verbose=False,
)

prompt = """<|im_start|>system
You are BlitzKode, an expert coding assistant. Write clean, efficient, and practical code.<|im_end|>
<|im_start|>user
Write a Python function to find the longest substring without repeating characters.<|im_end|>
<|im_start|>assistant
"""

result = llm(
    prompt,
    max_tokens=256,
    temperature=0.3,
    stop=["<|im_end|>", "<|im_start|>user"],
)

print(result["choices"][0]["text"].strip())
```

### `llama.cpp` CLI

```bash
huggingface-cli download sajadkoder/blitzkode blitzkode.gguf --local-dir ./blitzkode

./llama-cli \
  -m ./blitzkode/blitzkode.gguf \
  -p "Write a Python function that checks whether a linked list has a cycle." \
  --ctx-size 2048 \
  --temp 0.3 \
  -n 256
```

### Local server

The included FastAPI server in this repository can be started with:

```bash
python server.py
```

Then open `http://localhost:7860`.

## Prompt Format

The serving code in this repository uses a ChatML-style prompt template:

```text
<|im_start|>system
You are BlitzKode, an AI coding assistant created by Sajad. You are an expert in Python, JavaScript, Java, C++, and other programming languages. Write clean, efficient, and well-documented code. Keep responses concise and practical.<|im_end|>
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
```

## License

BlitzKode is licensed under **MIT License**. See the project README for details.

When using this model, also comply with the upstream Qwen base model license.

## Contact

- GitHub: https://github.com/sajadkoder/blitzkode
- Issues and contributions welcome!
