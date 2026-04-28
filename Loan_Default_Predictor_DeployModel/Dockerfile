# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Copy source code ─────────────────────────────────────────────────────────
COPY app.py model.py drift.py explain.py ./

# ─── Copy data (for inference + drift reference) ──────────────────────────────
COPY data/ ./data/

# ─── Create artifact + model directories ─────────────────────────────────────
RUN mkdir -p artifacts models mlruns

# ─── Environment variables ────────────────────────────────────────────────────
ENV MODEL_PATH=models/model_Exp1_Conservative_XGB.pkl
ENV MLFLOW_TRACKING_URI=./mlruns
ENV PORT=5000

# ─── Expose Flask port ────────────────────────────────────────────────────────
EXPOSE 5000

# ─── Default: run Flask API ──────────────────────────────────────────────────
#     To train instead: docker run <image> python model.py
CMD ["python", "app.py"]