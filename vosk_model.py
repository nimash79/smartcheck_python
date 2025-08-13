from vosk import Model, KaldiRecognizer
import wave
import json
import sys

wav_path = "nima.wav"
model_path = "./models/vosk-model-fa-0.42"

# Load model
model = Model(model_path)
wf = wave.open(wav_path, "rb")

# Make sure it's mono PCM 16-bit, 16000 Hz
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 44100]:
    sys.exit("Audio file must be WAV format mono PCM.")

rec = KaldiRecognizer(model, wf.getframerate())

# Read chunks and transcribe
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print(result.get("text", ""))
    else:
        partial = json.loads(rec.PartialResult())
        print(partial.get("partial", ""), end="\r")

# Print final result
final_result = json.loads(rec.FinalResult())
print("\nFinal transcription:", final_result.get("text", ""))
