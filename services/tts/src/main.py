import os
import logging
import tempfile
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, AsyncGenerator
import io
import base64

from .voice_manager import VoiceManager, VoiceConfig, VoiceMode
from .audio_utils import AudioProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-TTS-12Hz-0.6B-Base")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/models/Qwen3-TTS-Tokenizer-12Hz")
PORT = int(os.getenv("PORT", 8002))
MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", 30))

app = FastAPI(
    title="Aether TTS Service",
    description="Qwen3-TTS-12Hz Enterprise Voice Synthesis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
tts_engine = None
voice_manager = None
audio_processor = None

class TTSRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "Vivian"  # Can be preset name or "cloned_XXX"
    mode: VoiceMode = "customvoice"
    response_format: Literal["mp3", "wav", "pcm", "base64"] = "base64"
    speed: float = 1.0
    emotion: Optional[str] = "neutral"
    language: Optional[str] = None  # Auto-detect if None
    voice_description: Optional[str] = None  # For VoiceDesign mode

@app.on_event("startup")
async def startup_event():
    """Initialize TTS Engine and Voice Manager"""
    global tts_engine, voice_manager, audio_processor
    
    logger.info("🔊 Initializing TTS Service...")
    logger.info(f"TTS Model: {MODEL_PATH}")
    logger.info(f"Tokenizer: {TOKENIZER_PATH}")
    
    try:
        # Initialize audio processor
        audio_processor = AudioProcessor()
        
        # Initialize voice manager
        config_path = "/app/configs/default-speakers.json"
        voice_manager = VoiceManager(config_path)
        
        # Import Qwen3-TTS components
        from transformers import Qwen3ForConditionalGeneration, AutoProcessor
        
        # Load model and tokenizer
        logger.info("Loading TTS model (this may take 2-3 minutes)...")
        
        # Note: Actual implementation would use Qwen3-TTS specific loading
        # This is a simplified version - real implementation needs vLLM-Omni integration
        processor = AutoProcessor.from_pretrained(TOKENIZER_PATH)
        model = Qwen3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tts_engine = {
            "model": model,
            "processor": processor
        }
        
        logger.info("✅ TTS Service initialized successfully")
        logger.info(f"Available voices: {list(voice_manager.list_voices().keys())}")
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        # For development, we'll continue with mock
        logger.warning("Running in mock mode - TTS will return placeholder")
        tts_engine = {"mock": True}

def generate_speech_base(text: str, reference_audio: str) -> np.ndarray:
    """Base mode: Zero-shot voice cloning"""
    try:
        # This would use the actual Qwen3-TTS inference
        # Simplified for structure - real impl needs vLLM-Omni audio generation
        
        # 1. Encode text
        # 2. Process reference audio through speech tokenizer
        # 3. Generate audio tokens via vLLM
        # 4. Decode to waveform
        
        # Placeholder: return dummy audio
        duration = len(text.split()) * 0.5  # ~0.5s per word
        samples = int(duration * 24000)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 0.3
        
        return audio
        
    except Exception as e:
        logger.error(f"Base generation error: {e}")
        raise

def generate_speech_preset(text: str, voice_id: str, emotion: str) -> np.ndarray:
    """CustomVoice mode: Use preset speakers"""
    try:
        voice = voice_manager.get_voice(voice_id)
        if not voice:
            raise ValueError(f"Unknown voice: {voice_id}")
        
        # Emotion control would modify the generation parameters
        # Placeholder implementation
        duration = len(text.split()) * 0.5
        samples = int(duration * 24000)
        
        # Different base frequencies for different voices (placeholder logic)
        base_freq = 440 if "female" in voice.description.lower() else 220
        audio = np.sin(2 * np.pi * base_freq * np.linspace(0, duration, samples)) * 0.3
        
        return audio
        
    except Exception as e:
        logger.error(f"Preset generation error: {e}")
        raise

def generate_speech_design(text: str, description: str) -> np.ndarray:
    """VoiceDesign mode: Create voice from description"""
    try:
        # Parse description for voice characteristics
        # Generate latent voice embedding
        # Synthesize
        
        duration = len(text.split()) * 0.5
        samples = int(duration * 24000)
        audio = np.sin(2 * np.pi * 330 * np.linspace(0, duration, samples)) * 0.3
        
        return audio
        
    except Exception as e:
        logger.error(f"VoiceDesign error: {e}")
        raise

@app.get("/health")
async def health_check():
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    return {
        "status": "healthy",
        "model": "Qwen3-TTS-12Hz",
        "voices_available": len(voice_manager.list_voices()) if voice_manager else 0
    }

@app.get("/v1/voices")
async def list_voices(mode: Optional[VoiceMode] = None):
    """List available voices"""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    
    voices = voice_manager.list_voices(mode)
    return {
        "object": "list",
        "data": [
            {
                "voice_id": name,
                "name": config.name,
                "mode": config.mode,
                "description": config.description,
                "preview_url": None  # Could serve sample clips
            }
            for name, config in voices.items()
        ]
    }

@app.post("/v1/audio/speech")
async def create_speech(
    input: str = Form(..., description="Text to synthesize"),
    voice: str = Form("Vivian", description="Voice ID or name"),
    mode: VoiceMode = Form("customvoice", description="Synthesis mode"),
    response_format: str = Form("base64", description="Output format"),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    emotion: Optional[str] = Form("neutral"),
    language: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None, description="For VoiceDesign mode"),
    reference_audio: Optional[UploadFile] = File(None, description="For Base mode cloning")
):
    """
    Main TTS endpoint supporting all three modes:
    - base: Clone from reference_audio upload (3s zero-shot)
    - customvoice: Use preset voices (Vivian, Ryan, etc.)
    - voicedesign: Generate voice from voice_description text
    """
    try:
        logger.info(f"TTS Request: mode={mode}, voice={voice}, text_len={len(input)}")
        
        # Validate inputs based on mode
        if mode == "base" and reference_audio is None:
            raise HTTPException(status_code=400, detail="reference_audio required for base mode")
        
        if mode == "voicedesign" and not voice_description:
            raise HTTPException(status_code=400, detail="voice_description required for voicedesign mode")
        
        # Process based on mode
        if mode == "base":
            # Save and preprocess reference audio
            temp_ref = tempfile.mktemp(suffix=".wav")
            with open(temp_ref, "wb") as f:
                f.write(await reference_audio.read())
            
            processed_ref = audio_processor.preprocess_reference(temp_ref, max_duration=3)
            
            # Generate
            audio_array = generate_speech_base(input, processed_ref)
            
            # Cleanup
            os.remove(temp_ref)
            if os.path.exists(processed_ref):
                os.remove(processed_ref)
                
        elif mode == "customvoice":
            audio_array = generate_speech_preset(input, voice, emotion)
            
        elif mode == "voicedesign":
            audio_array = generate_speech_design(input, voice_description)
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        
        # Format output
        if response_format == "base64":
            base64_audio = audio_processor.encode_audio_base64(audio_array, sample_rate=24000)
            return JSONResponse(content={
                "object": "audio",
                "data": base64_audio,
                "format": "wav",
                "sample_rate": 24000,
                "duration": len(audio_array) / 24000
            })
        
        elif response_format in ["wav", "mp3"]:
            # Return binary audio file
            buffer = io.BytesIO()
            import soundfile as sf
            sf.write(buffer, audio_array, 24000, format=response_format.upper())
            buffer.seek(0)
            
            media_type = "audio/wav" if response_format == "wav" else "audio/mpeg"
            return StreamingResponse(buffer, media_type=media_type)
            
        elif response_format == "pcm":
            # Raw PCM bytes
            pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            return StreamingResponse(io.BytesIO(pcm_bytes), media_type="audio/pcm")
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {response_format}")
            
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/clone")
async def clone_voice(
    name: str = Form(..., description="Name for the cloned voice"),
    reference_audio: UploadFile = File(..., description="3-second reference audio"),
    description: Optional[str] = Form(None)
):
    """
    Endpoint to register a new cloned voice for reuse
    Returns voice_id for use in /v1/audio/speech
    """
    try:
        # Save reference
        temp_path = tempfile.mktemp(suffix=".wav")
        with open(temp_path, "wb") as f:
            f.write(await reference_audio.read())
        
        # Validate
        valid, msg = audio_processor.validate_audio_file(temp_path)
        if not valid:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=msg)
        
        # Register
        voice = voice_manager.register_cloned_voice(name, temp_path)
        
        return {
            "voice_id": name,
            "mode": "base",
            "description": description or f"Cloned voice: {name}",
            "status": "registered"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/v1/audio/stream")
async def websocket_tts(websocket):
    """
    WebSocket endpoint for streaming TTS
    Allows real-time synthesis with chunked text input
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            text = data.get("text")
            voice = data.get("voice", "Vivian")
            mode = data.get("mode", "customvoice")
            
            # Generate (simplified - real impl would chunk)
            if mode == "customvoice":
                audio = generate_speech_preset(text, voice, "neutral")
            else:
                audio = generate_speech_base(text, data.get("reference"))
            
            # Send back as base64 chunks
            base64_audio = audio_processor.encode_audio_base64(audio)
            
            await websocket.send_json({
                "type": "audio",
                "data": base64_audio,
                "done": True
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=PORT, log_level="info")
