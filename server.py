import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from resemblyzer import VoiceEncoder, preprocess_wav
import os, sys, subprocess, json, time
import asyncio
import threading
import websockets
import face_recognition
import librosa
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
WS_URL = os.getenv("WS_URL")
PORT = os.getenv("PORT")
VOICES_FOLDER = os.getenv("VOICES_FOLDER")
os.makedirs(VOICES_FOLDER, exist_ok=True)

recordings  = {}

def load_audio(file_path):
    wav, _ = librosa.load(file_path, sr=16000)
    return wav

encoder = VoiceEncoder()

def create_embedding(file_path):
    wav = load_audio(file_path)
    v = encoder.embed_utterance(wav)
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    v = v / n
    return v

def record_audio(url, filename):
    """Start ffmpeg process to record audio stream"""
    output_path = os.path.join(VOICES_FOLDER, filename)

    cmd = [
        "ffmpeg", "-y", "-i", url,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


    recordings[url] = {"process": process, "file": output_path}
    process.wait()


app = Flask(__name__)
CORS(app)

def run_async_loop(coro):
    asyncio.run(coro)

def notify_node_server(data):
    async def send_data():
        try:
            async with websockets.connect(WS_URL) as websocket:
                await websocket.send(json.dumps(data))
        except Exception as e:
            print(f"[WebSocket] Error: {e}")

    threading.Thread(target=run_async_loop, args=(send_data(),)).start()

def get_faces():
    url = f"{API_URL}/persons/faces"
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            persons = response.json()["data"]["persons"]
            print("✅ Faces updated:", len(persons))
            return persons
        else:
            print("❌ Failed to fetch faces. Status:", response.status_code)
            return None
    except Exception as e:
        print("⚠️ Error while fetching faces:", e)
        return None

def generate_stream(camera_url, detection):
    persons = get_faces()
    COOLDOWN_SECONDS = 10
    last_seen = time.time() - COOLDOWN_SECONDS
    # detector = FaceDetector(minDetectionCon=0.8)

    while True:
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f"⚠️ Failed to connect to camera {camera_url}. Retrying in 3s...")
            time.sleep(3)
            continue
        print(f"✅ Connected to camera {camera_url}")

        while True:
            success, frame = cap.read()
            if not success:
                print(f"⚠️ Lost connection to camera {camera_url}. Reconnecting...")
                cap.release()
                time.sleep(2)
                break

            frame = cv2.resize(frame, (640, 480))
            now = time.time()

            if detection and now - last_seen >= COOLDOWN_SECONDS:
                last_seen = now
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img_rgb)

                if encodings:
                    face_data = {
                        "event": "face_detected",
                        "camera_url": camera_url,
                        "face": encodings[0].tolist(),
                        "exists": False,
                    }

                    if len(persons) > 1:
                        distances = face_recognition.face_distance(
                            [person["face"] for person in persons if person["face"] is not None], encodings[0]
                        )
                        min_distance = np.min(distances)
                        if min_distance <= 0.5:
                            face_data["exists"] = True

                    notify_node_server(face_data)
            # else:
            #   frame, _ = detector.findFaces(frame, draw=True)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/detection')
def detection():
    camera_url = request.args.get("camera_url")
    return Response(generate_stream(camera_url, True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    camera_url = request.args.get("camera_url")
    return Response(generate_stream(camera_url, False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route("/start_recording", methods=["GET"])
def start_recording():
    url = request.args.get("url")
    filename = f"rec_{int(time.time())}.wav"

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    thread = threading.Thread(target=record_audio, args=(url, filename))
    thread.start()

    return jsonify({"message": "Recording started", "file": filename})

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    url = request.json.get("url")

    process = recordings[url]["process"]
    process.terminate()
    file = recordings[url]["file"]
    del recordings[url]
    
    embedding = create_embedding(file)
    voice_data = {
        "event": "add_voice",
        "microphone_url": url,
        "voice": embedding.tolist()
    }
    notify_node_server(voice_data)
    
    os.remove(file)

    return jsonify({"message": "Recording stopped", "file": file})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
