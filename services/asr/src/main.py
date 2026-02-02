import os
import logging
import tempfile
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import soundfile as sf
import librosa
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-ASR-1.7B")
PORT = int(os.getenv("PORT", 8001))
VLLM_WORKERS = int(os.getenv("VLLM_WORKERS", 1))
MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", 32768))
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", 0.85))

app = FastAPI(
    title="Aether ASR Service",
    description="Qwen3-ASR-1.7B Automatic Speech Recognition",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
llm_engine = None
tokenizer = None

class TranscriptionRequest(BaseModel):
    model: str = "qwen3-asr"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Literal["json", "text", "verbose_json"] = "json"
    temperature: float = 0.0
    timestamp_granularities: Optional[list] = None

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM engine on startup"""
    global llm_engine, tokenizer
    
    logger.info(f"🎙️ Initializing ASR Service...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Initialize vLLM engine
        llm_engine = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=VLLM_WORKERS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="auto",
            device="cuda"
        )
        
        # Warm up
        logger.info("Warming up model...")
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        # Dummy inference to warm up
        dummy_audio = torch.zeros(16000)  # 1 second of silence
        # Note: Actual warm-up would need proper audio format
        
        logger.info("✅ ASR Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio to format expected by Qwen3-ASR
    Returns path to processed audio file
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample to 16kHz if needed (Qwen3-ASR expects 16kHz)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Save to temp file
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, 16000)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Audio preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": "Qwen3-ASR-1.7B",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """
    OpenAI-compatible audio transcription endpoint
    """
    from vllm import SamplingParams
    
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate file
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Supported: wav, mp3, m4a, flac, ogg"
        )
    
    temp_input = None
    temp_processed = None
    
    try:
        # Save uploaded file
        temp_input = tempfile.mktemp(suffix=os.path.splitext(file.filename)[1])
        with open(temp_input, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing audio file: {file.filename}, size: {len(content)} bytes")
        
        # Preprocess audio
        temp_processed = preprocess_audio(temp_input)
        
        # Get audio duration
        audio, sr = librosa.load(temp_processed, sr=16000)
        duration = len(audio) / 16000
        
        # Build conversation for Qwen3-ASR
        # Qwen3-ASR expects audio in the user message
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant that transcribes audio accurately."
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"file://{temp_processed}"}},
                    {"type": "text", "text": prompt or "Transcribe this audio."}
                ]
            }
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
        )
        
        outputs = llm_engine.generate(text, sampling_params)
        
        transcription = outputs[0].outputs[0].text.strip()
        
        # Clean up common artifacts
        transcription = transcription.replace("<|im_end|>", "").strip()
        
        logger.info(f"Transcription completed. Duration: {duration:.2f}s, Text length: {len(transcription)}")
        
        if response_format == "text":
            return TranscriptionResponse(text=transcription)
        
        return TranscriptionResponse(
            text=transcription,
            language=language or "auto",
            duration=duration
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp files
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_processed and os.path.exists(temp_processed):
            os.remove(temp_processed)

@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
    """
    OpenAI-compatible chat completion with audio input support
    """
    from vllm import SamplingParams
    
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.0)
        max_tokens = request.get("max_tokens", 256)
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
        )
        
        outputs = llm_engine.generate(text, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        return {
            "id": "asr-chat-completion",
            "object": "chat.completion",
            "created": int(torch.cuda.Event(enable_timing=False).elapsed_time(torch.cuda.Event(enable_timing=False)) if torch.cuda.is_available() else 0),
            "model": request.get("model", "qwen3-asr"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )
