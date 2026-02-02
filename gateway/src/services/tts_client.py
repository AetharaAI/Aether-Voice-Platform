import httpx
import logging
from typing import Optional, Literal
from config import settings

logger = logging.getLogger(__name__)

VoiceMode = Literal["base", "customvoice", "voicedesign"]

class TTSClient:
    """Client for TTS service"""
    
    def __init__(self):
        self.base_url = settings.tts_url
        self.timeout = settings.tts_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def synthesize(
        self,
        text: str,
        voice: str = "Vivian",
        mode: VoiceMode = "customvoice",
        response_format: str = "base64",
        speed: float = 1.0,
        emotion: Optional[str] = "neutral",
        voice_description: Optional[str] = None,
        reference_audio: Optional[bytes] = None
    ) -> dict:
        """
        Request speech synthesis from TTS service
        """
        try:
            data = {
                "input": text,
                "voice": voice,
                "mode": mode,
                "response_format": response_format,
                "speed": str(speed),
                "emotion": emotion or "neutral"
            }
            
            if voice_description:
                data["voice_description"] = voice_description
            
            files = {}
            if reference_audio:
                files["reference_audio"] = ("reference.wav", reference_audio, "audio/wav")
            
            response = await self.client.post(
                f"{self.base_url}/v1/audio/speech",
                data=data,
                files=files if files else None
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"TTS request failed: {e}")
            raise
    
    async def list_voices(self) -> list:
        """Get available voices"""
        try:
            response = await self.client.get(f"{self.base_url}/v1/voices")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        await self.client.aclose()
