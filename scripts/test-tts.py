#!/usr/bin/env python3
"""
Test script for TTS Service
"""
import requests
import sys
import time
import base64

TTS_URL = "http://localhost:8002"

def test_health():
    print("🔍 Testing TTS Health...")
    try:
        r = requests.get(f"{TTS_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ TTS Healthy: {r.json()}")
            return True
        else:
            print(f"❌ TTS Unhealthy: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

def test_list_voices():
    print("\n📋 Testing Voice List...")
    try:
        r = requests.get(f"{TTS_URL}/v1/voices")
        if r.status_code == 200:
            voices = r.json().get('data', [])
            print(f"✅ Found {len(voices)} voices:")
            for v in voices[:5]:
                print(f"   - {v['voice_id']}: {v['description'][:50]}...")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_synthesis():
    print("\n🔊 Testing TTS Synthesis...")
    
    text = "Hello, this is a test of the Aether voice synthesis system."
    
    try:
        data = {
            'input': text,
            'voice': 'Vivian',
            'mode': 'customvoice',
            'response_format': 'base64',
            'emotion': 'cheerful'
        }
        
        start = time.time()
        r = requests.post(f"{TTS_URL}/v1/audio/speech", data=data)
        latency = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            audio_data = result.get('data')
            duration = result.get('duration', 0)
            
            # Save to file
            if audio_data:
                with open('/tmp/test_tts_output.wav', 'wb') as f:
                    f.write(base64.b64decode(audio_data))
                print(f"✅ Synthesis Success ({latency:.2f}s, {duration:.2f}s audio)")
                print(f"   Saved to: /tmp/test_tts_output.wav")
                return True
        else:
            print(f"❌ Synthesis Failed: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TTS Service Test Suite\n")
    
    if not test_health():
        sys.exit(1)
    
    test_list_voices()
    test_synthesis()
