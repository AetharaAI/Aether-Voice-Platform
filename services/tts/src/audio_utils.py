import io
import base64
import logging
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Utility class for audio preprocessing and postprocessing"""
    
    TARGET_SR = 16000  # Qwen3-TTS expects 16kHz
    MAX_DURATION = 30  # seconds
    
    @staticmethod
    def load_audio(input_data: Union[str, bytes], is_base64: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio from file path, bytes, or base64 string
        Returns: (audio_array, sample_rate)
        """
        try:
            if is_base64:
                # Decode base64
                audio_bytes = base64.b64decode(input_data)
                # Convert to numpy array
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                samples = np.array(audio_segment.get_array_of_samples())
                sr = audio_segment.frame_rate
                
                # Convert stereo to mono if needed
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                
                # Convert to float32
                audio = samples.astype(np.float32) / (2**15)
                
            elif isinstance(input_data, str):
                # File path or base64 string detection
                if input_data.startswith('data:audio'):
                    # Extract base64 from data URI
                    base64_data = input_data.split(',')[1]
                    return AudioProcessor.load_audio(base64_data, is_base64=True)
                else:
                    # File path
                    audio, sr = librosa.load(input_data, sr=None, mono=True)
                    
            elif isinstance(input_data, bytes):
                # Raw bytes
                audio_segment = AudioSegment.from_file(io.BytesIO(input_data))
                samples = np.array(audio_segment.get_array_of_samples())
                sr = audio_segment.frame_rate
                
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                    
                audio = samples.astype(np.float32) / (2**15)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    @staticmethod
    def preprocess_reference(audio_input: Union[str, bytes, np.ndarray], 
                           max_duration: int = MAX_DURATION) -> str:
        """
        Preprocess reference audio for voice cloning
        Returns path to processed temp file
        """
        try:
            if isinstance(audio_input, np.ndarray):
                audio = audio_input
                sr = AudioProcessor.TARGET_SR
            else:
                audio, sr = AudioProcessor.load_audio(audio_input)
            
            # Resample if needed
            if sr != AudioProcessor.TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=AudioProcessor.TARGET_SR)
            
            # Trim to max duration
            max_samples = max_duration * AudioProcessor.TARGET_SR
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                logger.info(f"Trimmed reference audio to {max_duration}s")
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Save to temp file
            temp_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_path, audio, AudioProcessor.TARGET_SR)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error preprocessing reference: {e}")
            raise

    @staticmethod
    def encode_audio_base64(audio_array: np.ndarray, sample_rate: int = 24000) -> str:
        """
        Encode audio array to base64 string (for JSON response)
        Qwen3-TTS outputs 24kHz audio
        """
        try:
            # Ensure correct shape
            if len(audio_array.shape) == 1:
                audio_array = audio_array.reshape(1, -1)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_array.T, sample_rate, format='WAV')
            buffer.seek(0)
            
            # Encode to base64
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"Error encoding audio: {e}")
            raise

    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        """Validate audio file format and duration"""
        try:
            info = sf.info(file_path)
            duration = info.duration
            
            if duration > AudioProcessor.MAX_DURATION:
                return False, f"Audio too long: {duration:.1f}s > {AudioProcessor.MAX_DURATION}s"
            
            return True, f"Valid audio: {duration:.2f}s, {info.samplerate}Hz"
            
        except Exception as e:
            return False, str(e)
