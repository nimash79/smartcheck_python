import os
import sys
import json
import subprocess
import threading
import requests
import websockets
import asyncio
import numpy as np
from resemblyzer import VoiceEncoder
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
WS_URL  = os.getenv("WS_URL")

SAMPLE_RATE = 16000
CHUNK_BYTES = 8000

encoder = VoiceEncoder()

persons = []

async def listen_for_voices():
    global persons
    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                print("🔗 Connected to voices WebSocket...")
                async for message in websocket:
                    data = json.loads(message)
                    if data["type"] == "new_voice":
                        person = data.get("person", None)
                        if person:
                            persons.append(person)
                            print(f"✅ voices updated: {len(persons)}")
        except Exception as e:
            print(f"⚠️ WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

def get_voices():
    global persons
    url = f"{API_URL}/persons/voices"
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            persons = response.json()["data"]["persons"]
            del persons[0]
            print("✅ voices updated:", len(persons))
        else:
            print("❌ Failed to fetch voices. Status:", response.status_code)
    except Exception as e:
        print("⚠️ Error while fetching voices:", e)

def add_alarm(data):
    url = f"{API_URL}/alarms/add"
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        requests.post(url, json=data, headers=headers)
    except Exception as e:
        print("⚠️ Error while adding report:", e)


def build_ffmpeg_cmd(url: str):
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", url,
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]

def load_vosk(model_path: str):
    from vosk import Model, KaldiRecognizer
    if not os.path.isdir(model_path):
        print(f"ERROR: Vosk model not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    return recognizer

def identify_speaker(wav_data, threshold=0.75):
    global voices
    emb = encoder.embed_utterance(wav_data)
    
    # Compute cosine similarities
    scores = {person["personId"]: np.inner(emb, person["voice"]) for person in persons}
    print(scores)
    best_match = max(scores, key=scores.get)
    
    if scores[best_match] > threshold:
        return best_match
    else:
        return None

def pcm16_to_float32(raw_bytes):
    """Convert PCM16 bytes to float32 numpy array for Resemblyzer."""
    audio = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
    return audio


def main():
    get_voices()
    print(persons)
    recognizer = load_vosk("./models/vosk-model-fa-0.42")
    ffmpeg_cmd = build_ffmpeg_cmd("http://192.168.0.133:8080/audio.wav")

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)

    while True:
        data = proc.stdout.read(CHUNK_BYTES)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            text = res.get("text", "").strip()
            if text:
                print("text:", text)
                wav_data = pcm16_to_float32(data)
                speaker = identify_speaker(wav_data)
                if "جنگ" in text:
                    add_alarm({
                        "personId": speaker,
                        "description": f"مکالمه مشکوک: {text}",
                    })

if __name__ == "__main__":
    main()
