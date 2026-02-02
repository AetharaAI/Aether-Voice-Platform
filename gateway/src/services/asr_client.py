import httpx
import logging
from typing import Optional, BinaryIO
from config import settings

logger = logging.getLogger(__name__)

class ASRClient:
    """Client for ASR service"""
    
    def __init__(self):
        self.base_url = settings.asr_url
        self.timeout = settings.asr_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def transcribe(
        self, 
        audio_file: BinaryIO, 
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> dict:
        """
        Send audio to ASR service for transcription
        """
        try:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}
            data = {}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            
            response = await self.client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"ASR request failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if ASR service is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        await self.client.aclose()
