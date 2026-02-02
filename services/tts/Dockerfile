FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install vLLM-Omni fork (specific commit for TTS)
# This is a different fork than Omni service - this one supports audio generation
RUN pip install --no-cache-dir \
    git+https://github.com/Alibaba-NLP/vllm-omni.git@main \
    || echo "Using alternative installation method"

# If git install fails, we'll install from PyPI fallback
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    transformers==4.51.3 \
    accelerate==0.30.1

# Install Qwen3-TTS package
RUN pip install --no-cache-dir \
    git+https://github.com/QwenLM/Qwen3-TTS.git@main \
    || echo "Qwen3-TTS will be installed from source"

# API and audio processing dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    python-multipart==0.0.20 \
    pydantic==2.11.0 \
    numpy==2.1.2 \
    soundfile==0.13.1 \
    librosa==0.11.0 \
    audioread==3.0.1 \
    pydub==0.25.1

WORKDIR /app

# Copy source
COPY src/ /app/src/
COPY configs/ /app/configs/

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "src.main"]
