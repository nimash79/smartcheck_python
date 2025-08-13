import subprocess
import numpy as np
import sounddevice as sd

url = "http://192.168.0.133:8080/audio.wav"


ffmpeg_cmd = [
    "ffmpeg",
    "-i", url,
    "-f", "s16le",
    "-ac", "1",
    "-ar", "44100",
    "pipe:1"
]

proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

samplerate = 44100
channels = 1
blocksize = 1024

stream = sd.OutputStream(samplerate=samplerate, channels=channels, blocksize=blocksize, dtype='int16')
stream.start()

try:
    while True:
        data = proc.stdout.read(blocksize*2)  # 2 bytes per sample
        if not data:
            break
        audio = np.frombuffer(data, dtype=np.int16)
        stream.write(audio)
except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
    proc.terminate()
