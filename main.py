import cv2
import threading
import time
from collections import defaultdict
import websockets
import asyncio
import json
import face_recognition
import numpy as np
import requests
import math
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
WS_URL  = os.getenv("WS_URL")


persons = []

last_seen = defaultdict(float)
COOLDOWN_SECONDS = 30

CLEANUP_INTERVAL = 15 * 60

def cleanup_last_seen():
    while True:
        now = time.time()
        keys_to_delete = []

        for key, last_time in list(last_seen.items()):
            if now - last_time > CLEANUP_INTERVAL:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del last_seen[key]
            # Optional: log cleanup
            # print(f"Removed stale entry: {key}")

        time.sleep(CLEANUP_INTERVAL)


async def listen_for_faces():
    global persons
    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                print("🔗 Connected to faces WebSocket...")
                async for message in websocket:
                    data = json.loads(message)
                    if data["type"] == "new_face":
                        person = data.get("person", None)
                        if person:
                            persons.append(person)
                            print(f"✅ Faces updated: {len(persons)}")
        except Exception as e:
            print(f"⚠️ WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

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

def add_report(data):
    url = f"{API_URL}/reports/add"
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        requests.post(url, json=data, headers=headers)
    except Exception as e:
        print("⚠️ Error while adding report:", e)


def distance_to_confidence(distance):
    confidence = 1 / (1 + math.exp((distance - 0.5) * 15))
    return round(confidence * 100, 2)


def capture_camera(camera_url):
    video_capture = cv2.VideoCapture(camera_url)

    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            continue
        
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        frame = cv2.resize(frame, (640, 480))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(img_rgb)

        if encodings:
            face_encoding = encodings[0]
            # matches = face_recognition.compare_faces([person["face"] for person in persons], face_encoding, tolerance=0.6)
            # if True in matches:
            #     index = matches.index(True)
            #     name = persons[index]["name"]
            #     print(f"I found {name}")
            distances = face_recognition.face_distance([person["face"] for person in persons], face_encoding)
            min_distance = np.min(distances)
            data = {
                "personId": 0,
                "camera_url": camera_url,
                "confidenceFactor": 0
            }

            now = time.time()
            if min_distance <= 0.5:
                best_match_index = np.argmin(distances)
                confidence = distance_to_confidence(min_distance)
                person_id = persons[best_match_index]["personId"]
                data["personId"] = person_id
                data["confidenceFactor"] = round(confidence, 2)
                key = (camera_url, person_id)
            else:
                key = (camera_url, "unknown")

            # Send report only if cooldown passed
            if now - last_seen[key] >= COOLDOWN_SECONDS:
                last_seen[key] = now
                threading.Thread(target=add_report, args=(data,), daemon=True).start()

def start_all_cameras():
    camera_urls = ["rtsp://admin:@192.168.0.197:554/live/ch00_0", "http://192.168.0.133:8080/video"]

    for camera_url in camera_urls:
        threading.Thread(target=capture_camera, args=(camera_url,), daemon=True).start()

if __name__ == '__main__':
    get_faces()
    start_all_cameras()
    threading.Thread(target=cleanup_last_seen, daemon=True).start()
    asyncio.run(listen_for_faces())
