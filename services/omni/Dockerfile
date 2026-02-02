FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_V1=0  # Critical: Disable V1 engine for audio support

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install specific vLLM branch for Qwen3-Omni
# This is the wangxiongts/vllm fork with qwen3_omni branch
RUN pip install --no-cache-dir \
    git+https://github.com/wangxiongts/vllm.git@qwen3_omni \
    || pip install --no-cache-dir vllm==0.9.2

# Core dependencies
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    transformers==4.51.3 \
    accelerate==0.30.1 \
    numpy==2.1.2 \
    scipy==1.15.2

# API and audio
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    websockets==15.0.1 \
    python-multipart==0.0.20 \
    pydantic==2.11.0 \
    soundfile==0.13.1 \
    librosa==0.11.0

WORKDIR /app

COPY entrypoint.sh /app/
COPY src/ /app/src/

RUN chmod +x /app/entrypoint.sh

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
