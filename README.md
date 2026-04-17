# BlitzKode

BlitzKode is a local AI coding assistant that runs entirely on your machine. It generates code in Python, JavaScript, Java, C++, and other languages through a web interface or API. The model is fine-tuned from Qwen2.5-1.5B and quantized to GGUF format for fast inference.

## Tech Stack

- Model: Qwen2.5-1.5B (fine-tuned, GGUF format)
- Backend: Python, FastAPI, uvicorn
- Inference: llama.cpp / llama-cpp-python
- Frontend: Vanilla HTML, CSS, JavaScript
- Training: HuggingFace Transformers, PEFT, TRL

## Features

- Local code generation without external API calls
- Real-time streaming responses (token-by-token)
- Web UI with dark theme, conversation history, copy-to-clipboard
- REST API with streaming (SSE) support
- Multi-language support: Python, JavaScript, Java, C++, TypeScript, SQL
- Conversation context across multiple turns
- Configurable via environment variables
- Optional API key authentication
- CPU and GPU inference support
- Docker support

## Prerequisites

- Python 3.9+
- GGUF model file (`blitzkode.gguf`)
- 4GB+ RAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/neuralbroker/blitzkode.git
cd blitzkode

# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
# Place blitzkode.gguf in the project root, or set BLITZKODE_MODEL_PATH
```

## Usage

Start the server:

```bash
python server.py
```

Open `http://localhost:7860` in your browser.

### Docker

```bash
docker build -t blitzkode .
docker run -p 7860:7860 -v ./blitzkode.gguf:/app/blitzkode.gguf blitzkode
```

### API Examples

```bash
# Generate code
curl -X POST http://localhost:7860/generate -H 'Content-Type: application/json' -d '{
  \"prompt\": \"Write a Python function to reverse a string\"
}'

# Stream tokens
curl -X POST http://localhost:7860/generate/stream -H 'Content-Type: application/json' -d '{
  \"prompt\": \"Binary search implementation in Python\"
}'

# With conversation history
curl -X POST http://localhost:7860/generate -H 'Content-Type: application/json' -d '{
  \"prompt\": \"Add error handling to that function\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"Write a Python function to reverse a string\"},
    {\"role\": \"assistant\", \"content\": \"def reverse_string(s): return s[::-1]\"}
  ]
}'

# Check server health
curl http://localhost:7860/health

# Get API info
curl http://localhost:7860/info
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Your question or request |
| messages | array | [] | Conversation history (last 8 messages) |
| temperature | float | 0.5 | Response randomness (0.0-2.0) |
| max_tokens | int | 256 | Maximum tokens to generate |
| top_p | float | 0.95 | Nucleus sampling threshold |
| top_k | int | 20 | Top-k sampling |
| repeat_penalty | float | 1.05 | Repetition penalty |

## Project Structure

```
blitzkode/
├── server.py              # FastAPI backend, main entry point
├── blitzkode.gguf         # Quantized model file (~3GB)
├── Dockerfile             # Docker container
├── requirements.txt       # Serving dependencies
├── requirements-training.txt  # Training dependencies
├── LICENSE                # MIT License
├── .env.example           # Environment variable template
├── frontend/
│   └── index.html         # Web UI (HTML/CSS/JS)
├── tests/
│   └── test_server.py     # HTTP endpoint tests
├── scripts/
│   ├── train_sft.py       # Supervised fine-tuning (LoRA)
│   ├── train_grpo.py      # Reward-based SFT continuation
│   ├── train_dpo.py       # Direct Preference Optimization
│   ├── export_gguf.py     # Merge checkpoints and export GGUF
│   └── test_inference.py  # Direct model inference test
├── checkpoints/           # Trained LoRA adapter checkpoints
├── exported/              # Merged model for GGUF export
├── datasets/
│   └── raw/               # Training datasets
├── models/                # Base model files
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── MODEL_CARD.md          # Model documentation
└── README.md              # This file
```

## Environment Variables

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| BLITZKODE_PORT | 7860 | Server port | 8080 |
| BLITZKODE_HOST | 0.0.0.0 | Server bind address | 127.0.0.1 |
| BLITZKODE_GPU_LAYERS | 0 | GPU layers (0=CPU only) | 35 |
| BLITZKODE_N_CTX | 2048 | Context window size | 4096 |
| BLITZKODE_THREADS | auto | CPU threads for inference | 8 |
| BLITZKODE_BATCH | 128 | Batch size for processing | 256 |
| BLITZKODE_WORKERS | 2 | Concurrent request workers | 4 |
| BLITZKODE_MODEL_PATH | blitzkode.gguf | Path to model file | /path/to/model.gguf |
| BLITZKODE_FRONTEND_PATH | frontend/index.html | Path to frontend file | ./ui.html |
| BLITZKODE_MAX_PROMPT_LENGTH | 4000 | Max prompt characters | 8000 |
| BLITZKODE_PRELOAD_MODEL | false | Load model on startup | true |
| BLITZKODE_CORS_ORIGINS | * | CORS origins (comma-separated) | http://localhost:3000 |
| BLITZKODE_API_KEY | empty | API key (empty=disabled) | my-secret-key |

## Tests

```bash
python -m unittest discover -s tests -v
```

## Contributing

Contributions are welcome. Open an issue first for major changes.

## License

MIT License. See LICENSE file for details.
