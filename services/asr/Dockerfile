FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install vLLM with audio support (stable version for ASR)
# Qwen3-ASR works with standard vLLM 0.9.2
RUN pip install --no-cache-dir \
    vllm==0.9.2 \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    transformers==4.51.3 \
    accelerate==0.30.1 \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    python-multipart==0.0.20 \
    pydantic==2.11.0 \
    numpy==2.1.2 \
    soundfile==0.13.1 \
    librosa==0.11.0

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the server
CMD ["python", "-m", "src.main"]
