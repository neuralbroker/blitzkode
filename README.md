# BlitzKode

An AI-powered coding assistant built by **Sajad** using fine-tuned LLM technology.

## Features

- **Smart Code Generation** - Write clean, efficient, and well-documented code
- **Multiple Languages** - Python, JavaScript, Java, C++, and more
- **Fast Inference** - Optimized llama.cpp backend for rapid responses
- **Modern UI** - Clean, professional interface inspired by ChatGPT
- **Dark Theme** - Easy on the eyes for long coding sessions
- **Mobile Responsive** - Works on desktop and mobile devices
- **Streaming Support** - Real-time token streaming via API

## Tech Stack

- **Model**: Qwen2.5-1.5B (fine-tuned)
- **Backend**: Python + llama.cpp + FastAPI
- **Frontend**: Vanilla HTML/CSS/JS
- **Training**: HuggingFace Transformers + PEFT (LoRA)

## Quick Start

### Run the Server

```bash
# Start the server
python server.py
```

Open **http://localhost:7860** in your browser.

### API Usage

```bash
# Generate code
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in python", "max_tokens": 200}'

# Stream response
curl -X POST http://localhost:7860/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in python"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/generate` | POST | Generate response |
| `/generate/stream` | POST | Stream response |
| `/health` | GET | Health check |
| `/info` | GET | API info |

## Project Structure

```
BlitzKode/
├── server.py              # Main backend server
├── blitzkode.gguf        # Quantized model
├── frontend/
│   └── index.html       # Web interface
├── scripts/
│   ├── train_max.py     # Training pipeline
│   ├── export_gguf.py   # Model export
│   └── ...
└── checkpoints/         # Trained model checkpoints
```

## System Prompt

BlitzKode is configured with the following traits:
- Expert in all programming languages
- Clean, efficient code with comments
- Concise explanations
- Practical solutions
- Created by Sajad

## Performance Optimizations

- 35 GPU layers
- 4096 context window
- Flash Attention
- Memory locking
- GZip compression

## Version

Current version: **1.3**

## Creator

Built with love by **Sajad**

## License

MIT License
