import os
import logging
import torch
import uvicorn
import tempfile
import numpy as np
import base64
import json
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit")
PORT = int(os.getenv("PORT", 8003))
MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", 32768))
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", 0.85))

app = FastAPI(
    title="Aether Omni Service",
    description="Qwen3-Omni-30B Voice-to-Voice AI Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
llm_engine = None
tokenizer = None

class OmniChatRequest(BaseModel):
    messages: list
    model: str = "qwen3-omni"
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True
    modalities: list = ["audio", "text"]

class Session:
    """Manages conversation session with Omni model"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = []
        self.created_at = asyncio.get_event_loop().time()
        self.last_active = self.created_at
        
    def add_message(self, role: str, content: Any):
        self.conversation_history.append({"role": role, "content": content})
        self.last_active = asyncio.get_event_loop().time()
        
    def get_context(self, max_turns: int = 10) -> list:
        """Get recent conversation context"""
        return self.conversation_history[-max_turns:]

# Session storage
sessions: Dict[str, Session] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the Omni vLLM engine"""
    global llm_engine, tokenizer
    
    logger.info("🧠 Initializing Omni Service...")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"AWQ Quantization: Enabled (4-bit)")
    
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - Omni requires GPU")
        
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Initialize LLM with AWQ quantization
        logger.info("Loading AWQ quantized model (this takes ~3-5 minutes)...")
        
        llm_engine = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            quantization="awq",  # Explicit AWQ loading
            tensor_parallel_size=1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="auto",
            device="cuda",
            enforce_eager=False,  # Enable CUDA graph for speed
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048
        )
        
        logger.info("✅ Omni Service initialized successfully")
        logger.info("Ready for voice-to-voice conversations")
        
    except Exception as e:
        logger.error(f"Failed to initialize Omni: {e}")
        # Don't crash, but mark as unavailable
        llm_engine = None

def process_audio_input(audio_data: str) -> str:
    """
    Process incoming base64 audio to format expected by model
    Returns path to temp audio file
    """
    try:
        # Decode base64
        if "," in audio_data:
            audio_data = audio_data.split(",")[1]
            
        audio_bytes = base64.b64decode(audio_data)
        
        # Save to temp file
        temp_path = tempfile.mktemp(suffix=".wav")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
            
        # Verify/convert to 16kHz mono (Omni expects this)
        import librosa
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)
        sf.write(temp_path, audio, 16000)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise

def audio_to_base64(audio_array: np.ndarray, sr: int = 24000) -> str:
    """Convert numpy audio array to base64 string"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sr, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

@app.get("/health")
async def health_check():
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Omni model not loaded")
    
    # Check VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
    else:
        allocated = reserved = 0
    
    return {
        "status": "healthy",
        "model": "Qwen3-Omni-30B-A3B-AWQ-4bit",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_allocated_gb": round(allocated, 2),
        "vram_reserved_gb": round(reserved, 2),
        "active_sessions": len(sessions)
    }

@app.post("/v1/chat/completions")
async def chat_completion(request: OmniChatRequest):
    """
    OpenAI-compatible chat completions with audio support
    Supports multimodal input (audio + text) and audio output
    """
    from vllm import SamplingParams
    
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Process messages - handle audio content
        processed_messages = []
        temp_files = []
        
        for msg in request.messages:
            if isinstance(msg.get("content"), list):
                # Multimodal content
                new_content = []
                for item in msg["content"]:
                    if item.get("type") == "audio_url":
                        # Process base64 audio
                        audio_path = process_audio_input(item["audio_url"]["url"])
                        temp_files.append(audio_path)
                        new_content.append({
                            "type": "audio",
                            "audio": f"file://{audio_path}"
                        })
                    else:
                        new_content.append(item)
                processed_messages.append({
                    "role": msg["role"],
                    "content": new_content
                })
            else:
                processed_messages.append(msg)
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Configure generation
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_token_ids=[
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|audio|>")
            ]
        )
        
        # Generate
        outputs = llm_engine.generate(prompt, sampling_params)
        
        # Process output
        text_output = outputs[0].outputs[0].text
        
        # Extract audio if present (Omni outputs audio tokens that need decoding)
        # This is simplified - real implementation would decode audio tokens
        audio_output = None
        
        response = {
            "id": f"omni-{asyncio.get_event_loop().time()}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_output,
                    "audio": audio_output  # Base64 encoded if present
                },
                "finish_reason": "stop"
            }]
        }
        
        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/v1/audio/chat")
async def websocket_voice_chat(websocket: WebSocket, session_id: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time voice-to-voice conversation
    Handles full-duplex audio streaming
    """
    await websocket.accept()
    
    # Get or create session
    if session_id is None or session_id not in sessions:
        session_id = f"session_{asyncio.get_event_loop().time()}"
        sessions[session_id] = Session(session_id)
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Resuming session: {session_id}")
    
    session = sessions[session_id]
    
    try:
        while True:
            # Receive audio chunk or command
            message = await websocket.receive_json()
            
            msg_type = message.get("type")
            
            if msg_type == "audio":
                # Process incoming audio (base64)
                audio_b64 = message.get("data")
                audio_path = process_audio_input(audio_b64)
                
                # Add to conversation
                session.add_message("user", [
                    {"type": "audio", "audio_url": {"url": f"file://{audio_path}"}}
                ])
                
                # Generate response
                from vllm import SamplingParams
                
                # Build context
                context = session.get_context()
                prompt = tokenizer.apply_chat_template(
                    context,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=1024,
                    stop_token_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>")]
                )
                
                # Generate (non-streaming for simplicity in WS)
                outputs = llm_engine.generate(prompt, sampling_params)
                response_text = outputs[0].outputs[0].text.strip()
                
                # Extract audio output (mock for now)
                # Real: decode audio tokens from model output
                duration = len(response_text.split()) * 0.5
                samples = int(duration * 24000)
                audio_array = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 0.3
                audio_b64_out = audio_to_base64(audio_array)
                
                # Store assistant response
                session.add_message("assistant", response_text)
                
                # Send back
                await websocket.send_json({
                    "type": "response",
                    "session_id": session_id,
                    "text": response_text,
                    "audio": audio_b64_out,
                    "done": True
                })
                
                # Cleanup
                os.remove(audio_path)
                
            elif msg_type == "text":
                # Text-only message (for testing)
                text = message.get("data")
                session.add_message("user", text)
                
                # Generate
                from vllm import SamplingParams
                context = session.get_context()
                prompt = tokenizer.apply_chat_template(
                    context,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
                outputs = llm_engine.generate(prompt, sampling_params)
                response_text = outputs[0].outputs[0].text.strip()
                
                session.add_message("assistant", response_text)
                
                await websocket.send_json({
                    "type": "response",
                    "session_id": session_id,
                    "text": response_text,
                    "audio": None,
                    "done": True
                })
                
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                
            elif msg_type == "reset":
                # Reset conversation
                session.conversation_history = []
                await websocket.send_json({
                    "type": "status",
                    "message": "Conversation reset"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Don't delete session immediately - allow reconnection
        pass

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clean up a conversation session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=PORT, log_level="info")
