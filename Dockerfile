# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf \
    TRANSFORMERS_CACHE=/hf \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8501 \
    MODEL_PATH=/app/Mistral-7B-Instruct-v0.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app
RUN pip install -r requirements.txt

# --- Hugging Face Model Download Modifications ---

# Authenticate and download the gated model
# Use the environment variable for login
RUN --mount=type=secret,id=huggingface_token \
    pip install -U "huggingface_hub[cli]" && \
    python -m hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir "/app/Mistral-7B-Instruct-v0.3" \
    --local-dir-use-symlinks False

# --- Copy your app code ---
COPY src/*.py /app/

RUN mkdir -p /hf /tmp/offload && chmod -R 777 /hf /tmp/offload

# --- Expose and run ---
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]