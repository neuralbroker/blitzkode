# BlitzKode

Local coding assistant. Fine-tuned Qwen2.5-1.5B in GGUF format, served over HTTP.

Exists because I needed code generation that runs on my hardware without calling external APIs.

## Install

```bash
pip install llama-cpp-python fastapi uvicorn pydantic
```

Model file (`blitzkode.gguf`) must exist. If not, see `scripts/export_gguf.py`.

## Run

```bash
python server.py
```

Open `http://localhost:7860`.

## API

```bash
# Generate
curl -X POST http://localhost:7860/generate -H 'Content-Type: application/json' -d '{
  \"prompt\": \"write hello world in python\"
}'

# Stream tokens
curl -X POST http://localhost:7860/generate/stream -H 'Content-Type: application/json' -d '{
  \"prompt\": \"binary search in python\"
}'
```

## Environment Variables

| Variable | Default | What |
|----------|---------|------|
| BLITZKODE_PORT | 7860 | Port |
| BLITZKODE_N_CTX | 2048 | Context length |
| BLITZKODE_THREADS | auto | CPU threads |
| BLITZKODE_MODEL_PATH | blitzkode.gguf | Model file |

## Files

- `server.py` - FastAPI app
- `frontend/index.html` - Browser UI
- `scripts/train_*.py` - Training stages (SFT, GRPO, DPO)
- `scripts/export_gguf.py` - Merges checkpoints and exports GGUF

## Training Pipeline

1. `train_sft.py` - LoRA fine-tuning on coding data
2. `train_grpo.py` - Reward-based continuation
3. `train_dpo.py` - Preference optimization
4. `export_gguf.py` - Merge + convert

Each stage produces a checkpoint. Checkpoints are not guaranteed to be reproducible with current scripts.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests hit the HTTP endpoints. No benchmark suite.

## Limits

- CPU only. GPU layers disabled.
- 2048 token context.
- No images. Text only.
- Model accuracy not evaluated against standard benchmarks.

## License

MIT