import cv2
from flask import Flask, Response, request
from cvzone.FaceDetectionModule import FaceDetector
import asyncio
import threading
import websockets
import json
import face_recognition
import time
import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
WS_URL = os.getenv("WS_URL")
PORT = os.getenv("PORT")


app = Flask(__name__)

def run_async_loop(coro):
    asyncio.run(coro)

def notify_node_server(face_data):
    async def send_face():
        try:
            async with websockets.connect(WS_URL) as websocket:
                await websocket.send(json.dumps(face_data))
        except Exception as e:
            print(f"[WebSocket] Error: {e}")

    threading.Thread(target=run_async_loop, args=(send_face(),)).start()

persons = []
def get_faces():
    global persons
    url = f"{API_URL}/persons/faces"
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            persons = response.json()["data"]["persons"]
            print("✅ Faces updated:", len(persons))
        else:
            print("❌ Failed to fetch faces. Status:", response.status_code)
    except Exception as e:
        print("⚠️ Error while fetching faces:", e)

    threading.Timer(3600, get_faces).start()

def generate_stream(camera_url, detection):
    get_faces()
    
    COOLDOWN_SECONDS = 10
    last_seen = time.time() - COOLDOWN_SECONDS
    
    # cap = cv2.VideoCapture(f"http://{camera_ip}:8080/video")
    # cap = cv2.VideoCapture("rtsp://admin:@192.168.0.197:554/live/ch00_0")
    cap = cv2.VideoCapture(camera_url)
    detector = FaceDetector(minDetectionCon=0.8)

    while True:
        success, frame = cap.read()
        if not success:
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
                    distances = face_recognition.face_distance([person["face"] for person in persons], encodings[0])
                    min_distance = np.min(distances)
                    if min_distance <= 0.5:
                        face_data["exists"] = True
                notify_node_server(face_data)
        # else:
        #     frame, _ = detector.findFaces(frame, draw=True)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
