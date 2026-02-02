#!/bin/bash

MODEL_BASE="/mnt/aetherpro-extra1/models/llm/qwen3"
REQUIRED_MODELS=(
    "Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    "Qwen3-TTS-12Hz-0.6B-Base"
    "Qwen3-TTS-Tokenizer-12Hz"
    "Qwen3-ASR-1.7B"
)

echo "🔍 Verifying Aether Voice Platform Model Files..."
echo "Base Path: $MODEL_BASE"
echo ""

all_present=true

for model in "${REQUIRED_MODELS[@]}"; do
    path="$MODEL_BASE/$model"
    if [ -d "$path" ]; then
        size=$(du -sh "$path" | cut -f1)
        echo "✅ $model ($size)"
        
        # Check for critical files
        if [[ "$model" == *"AWQ"* ]]; then
            if [ ! -f "$path/model.safetensors.index.json" ] && [ ! -f "$path/model.safetensors" ]; then
                echo "   ⚠️  Warning: No model weights found"
                all_present=false
            fi
        fi
        
        if [[ "$model" == *"TTS"* ]] && [[ "$model" != *"Tokenizer"* ]]; then
            if [ ! -d "$path/speech_tokenizer" ]; then
                echo "   ⚠️  Warning: speech_tokenizer directory missing"
            fi
        fi
    else
        echo "❌ $model - MISSING"
        all_present=false
    fi
done

echo ""
if [ "$all_present" = true ]; then
    echo "🎉 All required models present. Ready to build!"
    exit 0
else
    echo "⚠️  Some models are missing. Please download them first."
    exit 1
fi
