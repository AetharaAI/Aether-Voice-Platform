import logging
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from session_manager import SessionManager
from services.asr_client import ASRClient
from services.tts_client import TTSClient, VoiceMode
from services.omni_client import OmniClient

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
session_manager: SessionManager = None
asr_client: ASRClient = None
tts_client: TTSClient = None
omni_client: OmniClient = None

security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global session_manager, asr_client, tts_client, omni_client
    
    logger.info("🚀 Initializing API Gateway...")
    
    # Initialize clients
    session_manager = SessionManager(settings.redis_url, settings.session_ttl)
    asr_client = ASRClient()
    tts_client = TTSClient()
    omni_client = OmniClient()
    
    logger.info("✅ Gateway initialized")
    yield
    
    # Cleanup
    logger.info("🧹 Cleaning up...")
    await asr_client.close()
    await tts_client.close()

app = FastAPI(
    title="Aether Voice Gateway",
    description="Unified API for ASR, TTS, and Omni Voice Services",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for auth (optional)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if settings.jwt_secret and credentials:
        # Verify JWT here if configured
        pass
    return credentials

@app.get("/health")
async def health_check():
    """Composite health check of all services"""
    asr_healthy = await asr_client.health_check()
    tts_healthy = await tts_client.health_check()
    omni_healthy = await omni_client.health_check()
    
    status = "healthy" if all([asr_healthy, tts_healthy, omni_healthy]) else "degraded"
    
    return {
        "status": status,
        "gateway": "healthy",
        "services": {
            "asr": "healthy" if asr_healthy else "unavailable",
            "tts": "healthy" if tts_healthy else "unavailable",
            "omni": "healthy" if omni_healthy else "unavailable"
        },
        "version": "1.0.0"
    }

# ASR Routes
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("qwen3-asr"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    user: str = Depends(verify_token)
):
    """
    Speech-to-text transcription
    Proxies to ASR service
    """
    try:
        result = await asr_client.transcribe(
            audio_file=file.file,
            language=language,
            prompt=prompt
        )
        return result
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TTS Routes
@app.get("/v1/voices")
async def list_voices():
    """List available TTS voices"""
    voices = await tts_client.list_voices()
    return {"object": "list", "data": voices}

@app.post("/v1/audio/speech")
async def create_speech(
    input: str = Form(..., description="Text to synthesize"),
    voice: str = Form("Vivian"),
    mode: str = Form("customvoice"),
    response_format: str = Form("base64"),
    speed: float = Form(1.0),
    emotion: Optional[str] = Form("neutral"),
    voice_description: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None)
):
    """
    Text-to-speech synthesis
    Supports three modes: base, customvoice, voicedesign
    """
    try:
        ref_bytes = None
        if reference_audio:
            ref_bytes = await reference_audio.read()
        
        result = await tts_client.synthesize(
            text=input,
            voice=voice,
            mode=mode,
            response_format=response_format,
            speed=speed,
            emotion=emotion,
            voice_description=voice_description,
            reference_audio=ref_bytes
        )
        
        if response_format in ["wav", "mp3"]:
            # Stream binary audio
            import base64
            audio_data = base64.b64decode(result["data"])
            media_type = "audio/wav" if response_format == "wav" else "audio/mpeg"
            return StreamingResponse(
                iter([audio_data]),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=speech.{response_format}"}
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Omni Routes
@app.websocket("/v1/audio/chat")
async def websocket_voice_chat(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for voice-to-voice AI conversation
    Connects to Omni service for full-duplex audio
    """
    await websocket.accept()
    client_omni = OmniClient()
    
    try:
        # Connect to Omni service
        new_session_id = await client_omni.connect(session_id)
        
        # Send session info to client
        await websocket.send_json({
            "type": "session",
            "session_id": new_session_id,
            "status": "connected"
        })
        
        # Create tasks for bidirectional streaming
        async def client_to_omni():
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "audio":
                    await client_omni.send_audio(msg["data"])
                elif msg.get("type") == "text":
                    await client_omni.send_text(msg["data"])
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        
        async def omni_to_client():
            async for msg in client_omni.stream():
                await websocket.send_json(msg)
        
        # Run both directions concurrently
        await asyncio.gather(
            client_to_omni(),
            omni_to_client(),
            return_exceptions=True
        )
        
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await client_omni.close()

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    OpenAI-compatible chat completions with audio support
    Routes to Omni service for processing
    """
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.omni_url}/v1/chat/completions",
                json=request,
                timeout=60.0
            )
            return response.json()
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session Management
@app.post("/v1/sessions")
async def create_session(user_id: str = "anonymous"):
    """Create new conversation session"""
    session_id = session_manager.create_session(user_id)
    return {"session_id": session_id, "user_id": user_id}

@app.delete("/v1/sessions/{session_id}")
async def end_session(session_id: str):
    """End conversation session"""
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.get("/")
async def root():
    return {
        "service": "Aether Voice Gateway",
        "version": "1.0.0",
        "endpoints": {
            "asr": "/v1/audio/transcriptions",
            "tts": "/v1/audio/speech",
            "omni": "/v1/audio/chat (WebSocket)",
            "chat": "/v1/chat/completions"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level,
        workers=2
    )
