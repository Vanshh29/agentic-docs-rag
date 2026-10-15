FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU-only torch keeps the image small (sentence-transformers needs torch)
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 .

# Bake the embedding model into the image so a cold start does not download it.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY data/samples ./data/samples
EXPOSE 8000
# Hosts such as Render inject $PORT; --proxy-headers makes the rate limiter see the real client IP.
CMD ["sh", "-c", "exec uvicorn docrag.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
