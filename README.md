# Aether Voice Platform

Enterprise Voice AI Platform powered by Qwen3 models - ASR, TTS, and Omni voice-to-voice agent.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Aether Voice Gateway                      │
│                    (Port 8080 - Unified API)                     │
├──────────────┬──────────────┬──────────────────────────────────┤
│   ASR        │    TTS       │        Omni (Voice Agent)         │
│  Service     │  Service     │         Service                   │
│  :8001       │   :8002      │          :8003                    │
├──────────────┼──────────────┼──────────────────────────────────┤
│ Qwen3-ASR-   │ Qwen3-TTS-   │  Qwen3-Omni-30B-A3B-Instruct      │
│ 1.7B         │ 12Hz-0.6B    │  -AWQ-4bit                        │
│              │              │                                   │
│ • OpenAI     │ • 9 Preset   │  • Voice-to-Voice                 │
│   compatible │   voices     │  • 120s+ conversations            │
│ • 16kHz      │ • 3 modes:   │  • WebSocket duplex               │
│   input      │   Base/      │                                   │
│              │   Custom/    │                                   │
│              │   Design     │                                   │
└──────────────┴──────────────┴──────────────────────────────────┘
                              │
                    ┌─────────┴────────┐
                    │     Redis        │
                    │  (Session Store) │
                    └──────────────────┘
```

## Quick Start

### 1. Verify Models

```bash
bash scripts/verify-models.sh
```

Required models in `/mnt/aetherpro-extra1/models/llm/qwen3`:
- `Qwen3-ASR-1.7B`
- `Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen3-TTS-Tokenizer-12Hz`
- `Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to match your setup
```

### 3. Build & Start

```bash
make build
make start
```

### 4. Test Services

```bash
# Terminal 1 - Test ASR
python scripts/test-asr.py

# Terminal 2 - Test TTS  
python scripts/test-tts.py

# Terminal 3 - Test Omni
python scripts/test-omni.py
```

## API Endpoints

### ASR (Speech-to-Text)
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=en"
```

### TTS (Text-to-Speech)
```bash
# CustomVoice mode (preset speakers)
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=Hello world" \
  -F "voice=Vivian" \
  -F "mode=customvoice"

# Base mode (voice cloning)
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=Hello world" \
  -F "mode=base" \
  -F "reference_audio=@voice_sample.wav"

# VoiceDesign mode
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=Hello world" \
  -F "mode=voicedesign" \
  -F "voice_description=A deep, authoritative male voice"
```

### Omni (Voice-to-Voice Agent)
WebSocket: `ws://localhost:8080/v1/audio/chat`

```javascript
const ws = new WebSocket('ws://localhost:8080/v1/audio/chat');

ws.onopen = () => {
  // Send audio (base64)
  ws.send(JSON.stringify({
    type: 'audio',
    data: base64AudioString
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  // response.text - AI text response
  // response.audio - base64 audio response
};
```

## Commands

```bash
make build        # Build all Docker images
make start        # Start all services
make stop         # Stop all services
make logs         # View all logs
make logs-asr     # View ASR logs only
make logs-tts     # View TTS logs only
make logs-omni    # View Omni logs only
make status       # Check service status
make verify       # Verify model files
make clean        # Clean up everything
```

## Resource Usage

| Service | VRAM | Startup Time |
|---------|------|--------------|
| ASR | ~5GB | ~30s |
| TTS | ~8GB | ~60s |
| Omni | ~22GB | ~120s |
| **Total** | **~35GB** | **~3min** |

## Troubleshooting

### GPU Not Found
```bash
# Check nvidia-docker
nvidia-smi

# Verify Docker runtime
docker info | grep -i nvidia
```

### Model Loading Errors
```bash
# Verify model paths
make verify

# Check model files exist
ls -la /mnt/aetherpro-extra1/models/llm/qwen3/
```

### Out of Memory
- Reduce `VLLM_GPU_MEMORY_UTILIZATION` in .env
- Lower `VLLM_MAX_MODEL_LEN` for shorter contexts

## License

MIT - See individual model licenses for Qwen3 models.
