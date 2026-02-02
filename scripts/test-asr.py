#!/usr/bin/env python3
"""
Test script for ASR Service
Requires: pip install requests
"""
import requests
import sys
import time

ASR_URL = "http://localhost:8001"

def test_health():
    print("🔍 Testing ASR Health...")
    try:
        r = requests.get(f"{ASR_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ ASR Healthy: {r.json()}")
            return True
        else:
            print(f"❌ ASR Unhealthy: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ ASR Connection Failed: {e}")
        return False

def test_transcription():
    print("\n🎙️ Testing ASR Transcription...")
    
    # Create dummy audio (1 second sine wave)
    import numpy as np
    import io
    import soundfile as sf
    
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    try:
        files = {'file': ('test.wav', buffer, 'audio/wav')}
        data = {'language': 'en', 'prompt': 'Transcribe clearly'}
        
        start = time.time()
        r = requests.post(f"{ASR_URL}/v1/audio/transcriptions", files=files, data=data)
        latency = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            print(f"✅ Transcription Success ({latency:.2f}s)")
            print(f"   Text: {result.get('text', 'N/A')[:100]}...")
            return True
        else:
            print(f"❌ Transcription Failed: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ASR Service Test Suite\n")
    
    if not test_health():
        sys.exit(1)
    
    # Wait for model to be ready
    time.sleep(2)
    
    test_transcription()
