import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import server


class FakeLlama:
    init_calls = []
    call_history = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLlama.init_calls.append(kwargs)

    def __call__(self, prompt, stream=False, **kwargs):
        FakeLlama.call_history.append(
            {
                "prompt": prompt,
                "stream": stream,
                "kwargs": kwargs,
            }
        )
        if stream:
            return iter(
                [
                    {"choices": [{"text": "Hello"}]},
                    {"choices": [{"text": " world"}]},
                ]
            )

        return {"choices": [{"text": "  hello world  "}]}


class ServerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeLlama.init_calls.clear()
        FakeLlama.call_history.clear()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.frontend_dir = self.root / "frontend"
        self.frontend_dir.mkdir(parents=True, exist_ok=True)
        self.frontend_path = self.frontend_dir / "index.html"
        self.frontend_path.write_text("<html><body>BlitzKode</body></html>", encoding="utf-8")
        self.model_path = self.root / "blitzkode.gguf"

    async def make_client(self, with_model=True, api_key=""):
        if with_model:
            self.model_path.write_text("fake-model", encoding="utf-8")
        elif self.model_path.exists():
            self.model_path.unlink()

        settings = server.Settings(
            root_dir=self.root,
            model_path=self.model_path,
            frontend_path=self.frontend_path,
            preload_model=False,
            api_key=api_key,
        )

        patcher = patch.object(server.llama_cpp, "Llama", FakeLlama)
        patcher.start()
        self.addCleanup(patcher.stop)

        app = server.create_app(settings)
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_root_serves_frontend(self):
        client = await self.make_client()
        response = await client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("BlitzKode", response.text)

    async def test_health_reports_artifact_status(self):
        client = await self.make_client()
        response = await client.get("/health")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "healthy")
        self.assertFalse(payload["model_loaded"])
        self.assertTrue(payload["model_exists"])
        self.assertTrue(payload["frontend_exists"])

    async def test_health_degraded_when_model_missing(self):
        client = await self.make_client(with_model=False)
        response = await client.get("/health")
        payload = response.json()

        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["model_exists"])

    async def test_info_endpoint(self):
        client = await self.make_client()
        response = await client.get("/info")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["name"], "BlitzKode")
        self.assertEqual(payload["version"], "2.0")
        self.assertIn("endpoints", payload)

    async def test_generate_uses_stubbed_model(self):
        client = await self.make_client()
        response = await client.post(
            "/generate",
            json={"prompt": "  write a loop  ", "max_tokens": 64},
        )
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["response"], "hello world")
        self.assertEqual(len(FakeLlama.init_calls), 1)
        self.assertIn("<|im_start|>user\nwrite a loop", FakeLlama.call_history[0]["prompt"])

    async def test_generate_with_conversation_history(self):
        client = await self.make_client()
        response = await client.post(
            "/generate",
            json={
                "prompt": "fix the bug",
                "messages": [
                    {"role": "user", "content": "write a loop"},
                    {"role": "assistant", "content": "for i in range(10): ..."},
                ],
            },
        )
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        prompt_used = FakeLlama.call_history[0]["prompt"]
        self.assertIn("<|im_start|>user\nwrite a loop", prompt_used)
        self.assertIn("<|im_start|>assistant\nfor i in range(10): ...", prompt_used)

    async def test_generate_rejects_blank_prompts(self):
        client = await self.make_client()
        response = await client.post("/generate", json={"prompt": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Prompt is required")

    async def test_generate_rejects_long_prompts(self):
        client = await self.make_client()
        long_prompt = "x" * 5000
        response = await client.post("/generate", json={"prompt": long_prompt})

        self.assertEqual(response.status_code, 400)
        self.assertIn("too long", response.json()["error"])

    async def test_generate_returns_503_when_model_is_missing(self):
        client = await self.make_client(with_model=False)
        response = await client.post("/generate", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("Model not found", response.json()["error"])

    async def test_stream_endpoint_returns_sse_chunks(self):
        client = await self.make_client()
        response = await client.post("/generate/stream", json={"prompt": "stream please"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertIn('data: {"token": "Hello"}', response.text)
        self.assertIn('data: {"token": " world"}', response.text)
        self.assertIn("data: [DONE]", response.text)

    async def test_stream_503_when_model_missing(self):
        client = await self.make_client(with_model=False)
        response = await client.post("/generate/stream", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 503)

    async def test_api_key_rejects_invalid(self):
        client = await self.make_client(api_key="secret123")
        response = await client.post(
            "/generate",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer wrong"},
        )

        self.assertEqual(response.status_code, 401)

    async def test_api_key_accepts_valid(self):
        client = await self.make_client(api_key="secret123")
        response = await client.post(
            "/generate",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer secret123"},
        )

        self.assertEqual(response.status_code, 200)

    async def test_no_api_key_required_when_not_set(self):
        client = await self.make_client(api_key="")
        response = await client.post("/generate", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)

    async def test_cors_headers_present(self):
        client = await self.make_client()
        response = await client.options(
            "/generate",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
