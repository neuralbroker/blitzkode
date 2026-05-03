#!/usr/bin/env python3
"""
BlitzKode backend server.

Serves the bundled frontend and proxies prompts to a local GGUF model
through llama.cpp. Model is loaded lazily so the module stays importable
in tests and environments where the model artifact is not present yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import llama_cpp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

APP_NAME = "BlitzKode"
APP_VERSION = "2.0"
CREATOR = "Sajad"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "blitzkode.gguf"
DEFAULT_FRONTEND_PATH = ROOT_DIR / "frontend" / "index.html"
DEFAULT_CONTEXT = 2048
DEFAULT_MAX_PROMPT_LENGTH = 4000
DEFAULT_MAX_TOKENS = 512
STOP_TOKENS = ["<|im_end|>", "<|im_start|>user"]

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are BlitzKode, an AI coding assistant created by Sajad. "
    "You are an expert in Python, JavaScript, Java, C++, and other programming languages. "
    "Write clean, efficient, and well-documented code. Keep responses concise and practical.<|im_end|>"
)

logger = logging.getLogger("blitzkode")


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _validate_prompt(prompt: str, max_length: int) -> tuple[str, JSONResponse | None]:
    prompt = prompt.strip()
    if not prompt:
        return prompt, JSONResponse({"error": "Prompt is required"}, status_code=400)
    if len(prompt) > max_length:
        return prompt, JSONResponse(
            {"error": f"Prompt too long. Max {max_length} chars."},
            status_code=400,
        )
    return prompt, None


@dataclass(slots=True)
class Settings:
    root_dir: Path = ROOT_DIR
    model_path: Path = Path(os.getenv("BLITZKODE_MODEL_PATH", DEFAULT_MODEL_PATH))
    frontend_path: Path = Path(os.getenv("BLITZKODE_FRONTEND_PATH", DEFAULT_FRONTEND_PATH))
    host: str = os.getenv("BLITZKODE_HOST", "0.0.0.0")
    port: int = _int_from_env("BLITZKODE_PORT", 7860)
    n_gpu_layers: int = _int_from_env("BLITZKODE_GPU_LAYERS", 0)
    n_ctx: int = _int_from_env("BLITZKODE_N_CTX", DEFAULT_CONTEXT)
    n_threads: int = _int_from_env("BLITZKODE_THREADS", max(1, min(8, os.cpu_count() or 1)))
    n_batch: int = _int_from_env("BLITZKODE_BATCH", 128)
    max_prompt_length: int = _int_from_env("BLITZKODE_MAX_PROMPT_LENGTH", DEFAULT_MAX_PROMPT_LENGTH)
    preload_model: bool = _bool_from_env("BLITZKODE_PRELOAD_MODEL", default=False)
    cors_origins: str = os.getenv("BLITZKODE_CORS_ORIGINS", "http://localhost:7860")
    api_key: str = os.getenv("BLITZKODE_API_KEY", "")


class MessageItem(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str
    messages: list[MessageItem] = Field(default_factory=list)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=DEFAULT_MAX_TOKENS)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=200)
    repeat_penalty: float = Field(default=1.05, ge=0.8, le=2.0)


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm: "llama_cpp.Llama | None" = None
        self._init_lock = threading.Lock()
        self._load_time_seconds: float | None = None
        self._last_error: str | None = None
        self._busy: bool = False

    @property
    def model_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_exists(self) -> bool:
        return self.settings.model_path.exists()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def load_time_seconds(self) -> float | None:
        return self._load_time_seconds

    @property
    def busy(self) -> bool:
        return self._busy

    def load_model(self):
        if self._llm is not None:
            return self._llm

        with self._init_lock:
            if self._llm is not None:
                return self._llm

            if not self.model_exists:
                self._last_error = f"Model not found at {self.settings.model_path}"
                raise FileNotFoundError(self._last_error)

            start_time = time.perf_counter()
            try:
                self._llm = llama_cpp.Llama(
                    model_path=str(self.settings.model_path),
                    n_gpu_layers=self.settings.n_gpu_layers,
                    n_ctx=self.settings.n_ctx,
                    n_threads=self.settings.n_threads,
                    n_batch=self.settings.n_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    seed=-1,
                )
                self._load_time_seconds = time.perf_counter() - start_time
                self._last_error = None
                logger.info("Model loaded in %.2fs (gpu_layers=%d)", self._load_time_seconds, self.settings.n_gpu_layers)
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Model load failed: %s", exc)
                raise

        return self._llm

    def build_prompt(self, req: GenerateRequest) -> str:
        parts = [SYSTEM_PROMPT]
        for msg in req.messages:
            if msg.role in ("user", "assistant"):
                parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{req.prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _gen_params(self, req: GenerateRequest) -> dict:
        return dict(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=STOP_TOKENS,
        )

    def generate_once(self, req: GenerateRequest) -> dict[str, object]:
        llm = self.load_model()
        self._busy = True
        try:
            start = time.perf_counter()
            result = llm(self.build_prompt(req), **self._gen_params(req))
            response = result["choices"][0]["text"].strip()
            elapsed = time.perf_counter() - start
            logger.info("Generated %d chars in %.2fs", len(response), elapsed)
            return {"response": response, "creator": CREATOR, "model": APP_NAME, "version": APP_VERSION}
        finally:
            self._busy = False

    def _run_stream(self, req: GenerateRequest, out_q: queue.Queue):
        """Runs streaming inference in a worker thread, puts tokens into out_q."""
        try:
            llm = self.load_model()
            self._busy = True
            start = time.perf_counter()
            token_count = 0
            for token in llm(self.build_prompt(req), stream=True, **self._gen_params(req)):
                if not token.get("choices"):
                    continue
                text = token["choices"][0].get("text", "")
                if text:
                    token_count += 1
                    out_q.put(f"data: {json.dumps({'token': text})}\n\n")
            elapsed = time.perf_counter() - start
            logger.info("Streamed %d tokens in %.2fs", token_count, elapsed)
            out_q.put("data: [DONE]\n\n")
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            out_q.put(f"data: {json.dumps({'error': str(exc)})}\n\n")
        finally:
            self._busy = False
            out_q.put(None)


def _check_api_key(request: Request, settings: Settings) -> JSONResponse | None:
    if not settings.api_key:
        return None
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else auth
    
    # Timing-safe comparison (prevent timing attacks)
    import hmac
    if not hmac.compare_digest(token, settings.api_key):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 5, window_seconds: int = 60):
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._clients: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._cleanup_done = 0

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        
        # Cleanup old entries periodically (every 1000 requests)
        self._cleanup_done += 1
        if self._cleanup_done > 1000:
            self._cleanup_done = 0
            with self._lock:
                cutoff = now - self._window
                self._clients = {ip: [t for t in ts if t >= cutoff] for ip, ts in self._clients.items() if ts}
        
        with self._lock:
            timestamps = self._clients.get(client_ip, [])
            timestamps = [t for t in timestamps if now - t < self._window]
            if len(timestamps) >= self._max:
                return JSONResponse(
                    {"error": "Rate limit exceeded. Try again later."},
                    status_code=429,
                    headers={"Retry-After": str(self._window)},
                )
            timestamps.append(now)
            self._clients[client_ip] = timestamps
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 50_000):
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._max:
            return JSONResponse({"error": "Request body too large"}, status_code=413)
        return await call_next(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    model_service = ModelService(settings)
    model_lock = asyncio.Lock()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload_model:
            try:
                await asyncio.to_thread(model_service.load_model)
            except Exception:
                pass
        yield

    app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.model_service = model_service

    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

    if _bool_from_env("BLITZKODE_RATE_LIMIT", default=True):
        app.add_middleware(RateLimitMiddleware, max_requests=_int_from_env("BLITZKODE_RATE_LIMIT_MAX", 5))
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=_int_from_env("BLITZKODE_MAX_REQUEST_BYTES", 50_000))

    @app.get("/")
    async def root():
        if not settings.frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend file is missing.")
        return FileResponse(str(settings.frontend_path))

    @app.get("/health")
    async def health():
        status = "healthy"
        if not settings.frontend_path.exists() or not model_service.model_exists:
            status = "degraded"
        return JSONResponse({
            "status": status,
            "model_loaded": model_service.model_loaded,
            "model_path": str(settings.model_path),
            "model_exists": model_service.model_exists,
            "frontend_exists": settings.frontend_path.exists(),
            "version": APP_VERSION,
            "gpu_layers": settings.n_gpu_layers,
            "last_error": model_service.last_error,
            "busy": model_service.busy,
        })

    @app.post("/generate")
    async def generate(req: GenerateRequest, request: Request):
        auth_err = _check_api_key(request, settings)
        if auth_err:
            return auth_err

        prompt, err = _validate_prompt(req.prompt, settings.max_prompt_length)
        if err:
            return err

        async with model_lock:
            try:
                sanitized = req.model_copy(update={"prompt": prompt})
                payload = await asyncio.to_thread(model_service.generate_once, sanitized)
                return JSONResponse(payload)
            except FileNotFoundError as exc:
                return JSONResponse({"error": str(exc)}, status_code=503)
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest, request: Request):
        auth_err = _check_api_key(request, settings)
        if auth_err:
            return auth_err

        prompt, err = _validate_prompt(req.prompt, settings.max_prompt_length)
        if err:
            return err

        if not model_service.model_exists:
            return JSONResponse({"error": f"Model not found at {settings.model_path}"}, status_code=503)

        sanitized = req.model_copy(update={"prompt": prompt})

        async def _locked_stream():
            async with model_lock:
                token_q: queue.Queue = queue.Queue()
                thread = threading.Thread(
                    target=model_service._run_stream,
                    args=(sanitized, token_q),
                    daemon=True,
                )
                thread.start()
                # Use thread-safe queue.get() instead of deprecated get_running_loop()
                while True:
                    chunk = await asyncio.to_thread(token_q.get)
                    if chunk is None:
                        break
                    yield chunk

        return StreamingResponse(
            _locked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/info")
    async def info():
        return JSONResponse({
            "name": APP_NAME,
            "creator": CREATOR,
            "version": APP_VERSION,
            "status": "ready" if model_service.model_exists else "model-missing",
            "mode": f"{'GPU' if settings.n_gpu_layers > 0 else 'CPU'} (llama.cpp)",
            "gpu_layers": settings.n_gpu_layers,
            "context_window": settings.n_ctx,
            "model_loaded": model_service.model_loaded,
            "load_time_seconds": model_service.load_time_seconds,
            "busy": model_service.busy,
            "endpoints": {
                "generate": "POST /generate",
                "stream": "POST /generate/stream",
                "health": "GET /health",
                "info": "GET /info",
            },
        })

    return app


app = create_app()


def main() -> None:
    s = Settings()
    print(f"\n{'=' * 50}")
    print(f"{APP_NAME.upper()} v{APP_VERSION}")
    print(f"Creator: {CREATOR}")
    print(f"{'=' * 50}")
    print(f"Model:  {s.model_path}")
    print(f"GPU:    {s.n_gpu_layers} layers")
    print(f"Ctx:    {s.n_ctx} | Threads: {s.n_threads}")
    print(f"URL:    http://localhost:{s.port}\n")

    uvicorn.run(app, host=s.host, port=s.port, log_level="warning")


if __name__ == "__main__":
    main()
