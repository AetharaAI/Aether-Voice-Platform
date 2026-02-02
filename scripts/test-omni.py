#!/usr/bin/env python3
"""
Test script for Omni Service
"""
import websocket
import json
import threading
import time
import sys

OMNI_WS_URL = "ws://localhost:8003/v1/audio/chat"

def on_message(ws, message):
    data = json.loads(message)
    msg_type = data.get('type')
    
    if msg_type == 'session':
        print(f"✅ Connected: Session {data.get('session_id')}")
    elif msg_type == 'response':
        print(f"🤖 Response: {data.get('text', '')[:100]}...")
        print(f"   Audio: {'Yes' if data.get('audio') else 'No'}")
        print(f"   Done: {data.get('done')}")
    elif msg_type == 'error':
        print(f"❌ Error: {data.get('message')}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"🔌 Connection Closed: {close_status_code}")

def on_open(ws):
    print("🧠 Omni Connection Opened")
    
    # Send text message for testing (no audio needed for basic test)
    def run():
        time.sleep(1)
        ws.send(json.dumps({
            "type": "text",
            "data": "Hello, can you hear me? Please introduce yourself briefly."
        }))
        time.sleep(5)
        ws.close()
    
    threading.Thread(target=run).start()

def test_websocket():
    print("🧠 Testing Omni WebSocket...")
    print(f"Connecting to {OMNI_WS_URL}")
    
    ws = websocket.WebSocketApp(
        OMNI_WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    ws.run_forever()

def test_health():
    import requests
    print("🔍 Testing Omni Health...")
    try:
        r = requests.get("http://localhost:8003/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ Omni Healthy: {r.json()}")
            return True
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
    return False

if __name__ == "__main__":
    print("🚀 Omni Service Test Suite\n")
    
    if not test_health():
        sys.exit(1)
    
    try:
        import websocket
    except ImportError:
        print("Installing websocket-client...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client"])
        import websocket
    
    test_websocket()
