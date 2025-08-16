from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import json


def load_audio(file_path):
    wav, _ = librosa.load(file_path, sr=16000)
    return wav


encoder = VoiceEncoder()


def create_embedding(file_path):
    wav = load_audio(file_path)
    return encoder.embed_utterance(wav)

nima_emb = create_embedding("nima.wav")
fatemeh_emb = create_embedding("fatemeh.wav")


known_speakers = {
    "Nima": nima_emb,
    "Fatemeh": fatemeh_emb,
}


def identify_speaker(file_path, threshold=0.75):
    wav = load_audio(file_path)
    emb = encoder.embed_utterance(wav)
    
    # Compute cosine similarities
    scores = {name: np.inner(emb, known_emb) for name, known_emb in known_speakers.items()}
    print(scores)
    best_match = max(scores, key=scores.get)
    
    if scores[best_match] > threshold:
        return best_match
    else:
        return "Unknown"

# Example usage:
speaker = identify_speaker("nima2.wav")
print("Identified speaker:", speaker)
