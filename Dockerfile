FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU-only torch keeps the image small (sentence-transformers needs torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data/samples ./data/samples
EXPOSE 8000
CMD ["uvicorn", "docrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
