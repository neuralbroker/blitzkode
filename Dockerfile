FROM python:3.11-slim

RUN useradd --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY frontend/ frontend/

ENV BLITZKODE_HOST=0.0.0.0
ENV BLITZKODE_PORT=7860
ENV BLITZKODE_GPU_LAYERS=0
ENV BLITZKODE_THREADS=4

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

USER appuser

CMD ["python", "server.py"]