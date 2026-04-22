# ─── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY --from=builder /install /usr/local

COPY src/ ./src/

COPY data/      ./data/
COPY models/    ./models/

RUN python src/model.py

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "src/app.py"]