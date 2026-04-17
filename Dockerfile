FROM python:3.11-slim

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

CMD ["python", "server.py"]
