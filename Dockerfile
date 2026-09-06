# Optional container image. The primary supported path is a local venv (see README);
# this exists for anyone who prefers Docker. Ollama runs as a separate service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://ollama:11434

# RapidOCR (onnxruntime + opencv) needs these system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY app ./app
COPY data/samples ./data/samples
COPY scripts ./scripts
COPY eval ./eval

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.headless=true"]
