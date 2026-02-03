#!/bin/bash

# Critical environment setup for Qwen3-Omni
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU_DEVICE_ID:-0}

# Use default if MODEL_PATH not set
MODEL_PATH=${MODEL_PATH:-/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit}

echo "🧠 Starting Aether Omni Service..."
echo "Model: ${MODEL_PATH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "VLLM V1 Engine: Disabled (required for audio output)"

# Verify model exists (warn but don't exit - let the app handle it)
if [ ! -d "${MODEL_PATH}" ]; then
    echo "⚠️  Warning: Model path ${MODEL_PATH} does not exist"
    echo "Available models in /models:"
    ls -la /models/ 2>/dev/null || echo "Cannot list /models"
fi

# Start the service
python -m src.main
