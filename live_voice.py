import os, sys, json, subprocess, requests
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
SAMPLE_RATE = 16000
CHUNK_BYTES = 8000  # 0.25 s at 16 kHz, 16-bit mono

encoder = VoiceEncoder()
persons = []  # [{ "personId": int/str, "voice": np.ndarray (normalized, float32) }, ...]
keywords = []

def build_ffmpeg_cmd(url: str):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", url, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]

def load_vosk(model_path: str):
    if not os.path.isdir(model_path):
        print(f"ERROR: Vosk model not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec

def fetch_and_prepare_person_embeddings():
    """Fetch voices from API, drop missing, convert to float32 arrays and L2-normalize."""
    global persons
    url = f"{API_URL}/persons/voices"
    try:
        r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        raw = r.json()["data"]["persons"]
    except Exception as e:
        print(f"⚠️ Failed to fetch voices: {e}")
        raw = []

    cleaned = []
    for p in raw:
        v = p.get("voice", None)
        if v is None:
            continue
        # Ensure numpy float32 and unit L2 norm
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-12
        v = v / n
        cleaned.append({"personId": p.get("personId"), "voice": v})
    persons = cleaned
    print(f"✅ voices prepared: {len(persons)}")

def fetch_keywords():
    global keywords
    url = f"{API_URL}/settings"
    try:
        request = requests.get(url, headers={"Content-Type": "application/json"})
        data = request.json()["data"]["setting"]["keyWords"]
        keywords = data.split(",")
    except Exception as e:
        print(f"⚠️ Failed to fetch keywords: {e}")
        keywords = []

def pcm16_bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0

def identify_speaker_from_wav(wav_f32_mono_16k: np.ndarray, threshold=0.70):
    """Returns personId or None. Uses VAD via preprocess_wav for stability."""
    if len(wav_f32_mono_16k) < int(0.6 * SAMPLE_RATE):
        return None  # too short to be reliable
    # VAD + preemphasis
    wav_proc = preprocess_wav(wav_f32_mono_16k, SAMPLE_RATE)
    if len(wav_proc) < int(0.5 * SAMPLE_RATE):
        return None
    emb = encoder.embed_utterance(wav_proc).astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-12)

    if not persons:
        return None

    # cosine similarities (dot product because everything is unit-norm)
    sims = {p["personId"]: float(np.dot(emb, p["voice"])) for p in persons}
    best_id = max(sims, key=sims.get)
    best_score = sims[best_id]
    # debug:
    # print({k: round(v, 3) for k, v in sims.items()})
    return best_id if best_score >= threshold else None

def add_alarm(data):
    try:
        requests.post(f"{API_URL}/alarms/add", json=data, headers={"Content-Type":"application/json"}, timeout=5)
    except Exception as e:
        print("⚠️ Error while adding report:", e)

def main():
    fetch_and_prepare_person_embeddings()
    fetch_keywords()

    recognizer = load_vosk("./models/vosk-model-fa-0.42")
    ffmpeg_cmd = build_ffmpeg_cmd("http://192.168.0.133:8080/audio.wav")

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Buffer ALL audio until Vosk emits a final result
    utter_bytes = bytearray()

    try:
        while True:
            data = proc.stdout.read(CHUNK_BYTES)
            if not data:
                break
            utter_bytes.extend(data)

            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                text = (res.get("text") or "").strip()
                if text:
                    # Use the FULL utterance audio for speaker ID
                    wav = pcm16_bytes_to_float32(bytes(utter_bytes))
                    speaker = identify_speaker_from_wav(wav, threshold=0.70)
                    # Clear buffer for next utterance
                    utter_bytes.clear()

                    # Debug/inspect
                    # print(f"Speaker: {speaker} | Text: {text}")

                    # Check keywords
                    for keyword in keywords:
                        if keyword in text:
                            add_alarm({
                            "personId": speaker,
                            "description": f"مکالمه مشکوک: {text}",
                            })
                            break
                        
                        
                else:
                    # If no text, still reset the buffer (Vosk finalized silence/noise)
                    utter_bytes.clear()

    except KeyboardInterrupt:
        pass
    finally:
        if proc and proc.poll() is None:
            proc.terminate()

if __name__ == "__main__":
    main()
