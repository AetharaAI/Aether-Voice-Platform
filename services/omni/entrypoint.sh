#!/bin/bash

# Critical environment setup for Qwen3-Omni
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID:-0}

echo "🧠 Starting Aether Omni Service..."
echo "Model: ${MODEL_PATH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "VLLM V1 Engine: Disabled (required for audio output)"

# Verify model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ Error: Model path ${MODEL_PATH} does not exist"
    exit 1
fi

# Start the service
python -m src.main
