import websockets
import json
import logging
from typing import Optional, Callable, AsyncGenerator
from config import settings

logger = logging.getLogger(__name__)

class OmniClient:
    """WebSocket client for Omni service"""
    
    def __init__(self):
        self.base_url = settings.omni_url.replace("http://", "ws://")
        self.session_id: Optional[str] = None
        
    async def connect(self, session_id: Optional[str] = None):
        """Connect to Omni WebSocket"""
        uri = f"{self.base_url}/v1/audio/chat"
        if session_id:
            uri += f"?session_id={session_id}"
            
        self.websocket = await websockets.connect(uri)
        if not session_id:
            # Get session ID from first message
            msg = await self.websocket.recv()
            data = json.loads(msg)
            self.session_id = data.get("session_id")
        else:
            self.session_id = session_id
            
        return self.session_id
    
    async def send_audio(self, audio_base64: str):
        """Send audio input"""
        await self.websocket.send(json.dumps({
            "type": "audio",
            "data": audio_base64
        }))
    
    async def send_text(self, text: str):
        """Send text input"""
        await self.websocket.send(json.dumps({
            "type": "text",
            "data": text
        }))
    
    async def receive(self) -> dict:
        """Receive message from Omni"""
        msg = await self.websocket.recv()
        return json.loads(msg)
    
    async def stream(self) -> AsyncGenerator[dict, None]:
        """Stream responses"""
        try:
            while True:
                msg = await self.websocket.recv()
                yield json.loads(msg)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Omni connection closed")
    
    async def close(self):
        if self.websocket:
            await self.websocket.close()
    
    async def health_check(self) -> bool:
        """Check HTTP health endpoint"""
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.omni_url}/health")
                return response.status_code == 200
        except:
            return False
