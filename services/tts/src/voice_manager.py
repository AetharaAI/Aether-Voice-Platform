import os
import json
import logging
from typing import Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

VoiceMode = Literal["base", "customvoice", "voicedesign"]

@dataclass
class VoiceConfig:
    name: str
    mode: VoiceMode
    description: Optional[str] = None
    preset_id: Optional[str] = None
    reference_audio: Optional[str] = None

class VoiceManager:
    """
    Manages voice presets and cloning for TTS service
    Three modes:
    1. Base: Zero-shot cloning from 3s reference audio
    2. CustomVoice: 9 preset speakers with emotion control
    3. VoiceDesign: Create voices from text descriptions
    """
    
    # 9 preset speakers from Qwen3-TTS
    PRESET_SPEAKERS = {
        "Vivian": {
            "id": "Vivian",
            "description": "Energetic young female, expressive and clear",
            "default_emotion": "neutral"
        },
        "Ryan": {
            "id": "Ryan",
            "description": "Professional male, authoritative and calm",
            "default_emotion": "neutral"
        },
        "Emma": {
            "id": "Emma", 
            "description": "Warm female, friendly and approachable",
            "default_emotion": "neutral"
        },
        "James": {
            "id": "James",
            "description": "Deep male voice, mature and trustworthy",
            "default_emotion": "neutral"
        },
        "Sophia": {
            "id": "Sophia",
            "description": "Soft-spoken female, gentle and soothing",
            "default_emotion": "neutral"
        },
        "Michael": {
            "id": "Michael",
            "description": "Energetic male, dynamic and engaging",
            "default_emotion": "neutral"
        },
        "Olivia": {
            "id": "Olivia",
            "description": "Bright female, cheerful and optimistic",
            "default_emotion": "neutral"
        },
        "William": {
            "id": "William",
            "description": "Scholarly male, precise and articulate",
            "default_emotion": "neutral"
        },
        "Ava": {
            "id": "Ava",
            "description": "Youthful female, modern and casual",
            "default_emotion": "neutral"
        }
    }
    
    EMOTIONS = ["neutral", "cheerful", "excited", "sad", "angry", "fearful", "disgusted"]
    
    def __init__(self, config_path: Optional[str] = None):
        self.voices: Dict[str, VoiceConfig] = {}
        self.custom_clones: Dict[str, str] = {}  # name -> reference_audio_path
        
        # Load preset speakers
        for name, info in self.PRESET_SPEAKERS.items():
            self.voices[name] = VoiceConfig(
                name=name,
                mode="customvoice",
                description=info["description"],
                preset_id=info["id"]
            )
        
        # Load custom configs if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, path: str):
        """Load custom voice configurations"""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            for voice_data in config.get("custom_voices", []):
                name = voice_data["name"]
                self.voices[name] = VoiceConfig(
                    name=name,
                    mode=voice_data.get("mode", "base"),
                    description=voice_data.get("description"),
                    reference_audio=voice_data.get("reference_audio")
                )
                logger.info(f"Loaded custom voice: {name}")
        except Exception as e:
            logger.error(f"Error loading voice config: {e}")
    
    def get_voice(self, voice_id: str) -> Optional[VoiceConfig]:
        """Get voice configuration by ID"""
        return self.voices.get(voice_id)
    
    def register_cloned_voice(self, name: str, reference_path: str) -> VoiceConfig:
        """
        Register a new voice cloned from reference audio (Base mode)
        """
        if name in self.voices:
            logger.warning(f"Overwriting existing voice: {name}")
        
        voice = VoiceConfig(
            name=name,
            mode="base",
            reference_audio=reference_path,
            description=f"Cloned voice: {name}"
        )
        self.voices[name] = voice
        self.custom_clones[name] = reference_path
        logger.info(f"Registered cloned voice: {name}")
        return voice
    
    def list_voices(self, mode: Optional[VoiceMode] = None) -> Dict[str, VoiceConfig]:
        """List available voices, optionally filtered by mode"""
        if mode is None:
            return self.voices
        return {k: v for k, v in self.voices.items() if v.mode == mode}
    
    def validate_emotion(self, emotion: str) -> str:
        """Validate and normalize emotion string"""
        emotion = emotion.lower()
        if emotion in self.EMOTIONS:
            return emotion
        return "neutral"  # default fallback
