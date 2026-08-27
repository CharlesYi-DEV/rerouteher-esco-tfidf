FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    SENTENCE_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appuser /app
USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY --chown=appuser:appuser api /app/api
COPY --chown=appuser:appuser model /app/model

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
