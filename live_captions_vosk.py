import os
import sys
import json
import queue
import threading
import subprocess
import numpy as np

SAMPLE_RATE = 16000
CHUNK_BYTES = 8000

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
    try:
        from vosk_model import Model, KaldiRecognizer
    except Exception as e:
        print("ERROR: Vosk not installed. Run: pip install vosk", file=sys.stderr)
        raise
    if not os.path.isdir(model_path):
        print(
            f"ERROR: Vosk model not found at '{model_path}'.\n"
            "Download an offline model (e.g., 'vosk-model-small-en-us-0.15') from:\n"
            "https://alphacephei.com/vosk/models\n"
            f"Then extract it and pass the folder path via --model <path>, or place it at: {model_path}",
            file=sys.stderr
        )
        sys.exit(1)
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    return recognizer

def reader_thread(proc, recognizer, out_q, stop_event):
    # Continuously read from ffmpeg stdout and push recognized text (partial/final) to queue
    while not stop_event.is_set():
        data = proc.stdout.read(CHUNK_BYTES)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            try:
                res = json.loads(recognizer.Result())
                text = res.get("text", "").strip()
                if text:
                    out_q.put(("final", text))
            except Exception:
                pass
        else:
            try:
                res = json.loads(recognizer.PartialResult())
                partial = res.get("partial", "").strip()
                if partial:
                    out_q.put(("partial", partial))
            except Exception:
                pass

def run_gui_loop(out_q, stop_event):
    try:
        import tkinter as tk
    except Exception:
        return False  # GUI unavailable
    
    root = tk.Tk()
    root.title("Live Captions (Vosk)")
    root.geometry("900x200")
    root.configure(bg="#111111")
    
    label = tk.Label(
        root,
        text="Waiting for speech...",
        font=("DejaVu Sans", 20),
        fg="#f0f0f0",
        bg="#111111",
        wraplength=860,
        justify="left",
        anchor="w"
    )
    label.pack(fill="both", expand=True, padx=16, pady=16)
    
    latest_final = ""
    latest_partial = ""

    def poll_queue():
        nonlocal latest_final, latest_partial
        try:
            while True:
                typ, txt = out_q.get_nowait()
                if typ == "final":
                    latest_final = (latest_final + " " + txt).strip()
                    latest_partial = ""
                else:
                    latest_partial = txt
        except queue.Empty:
            pass

        display = latest_final
        if latest_partial:
            display = (latest_final + " " + latest_partial).strip()
        label.config(text=display if display else "Listening...")
        if not stop_event.is_set():
            root.after(80, poll_queue)

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(80, poll_queue)
    root.mainloop()
    return True

def console_loop(out_q, stop_event):
    latest_final = ""
    latest_partial = ""
    print("[Console Mode] Live captions will appear below. Press Ctrl+C to stop.\n")
    while not stop_event.is_set():
        try:
            typ, txt = out_q.get(timeout=0.2)
            if typ == "final":
                latest_final = (latest_final + " " + txt).strip()
                latest_partial = ""
                print(f"[FINAL] {txt}")
            else:
                latest_partial = txt
                # Show the partial line without adding a newline each time
                sys.stdout.write(f"\r[PARTIAL] {latest_final} {latest_partial}   ")
                sys.stdout.flush()
        except queue.Empty:
            continue
    print("\nStopped.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live captions from phone audio via FFmpeg + Vosk")
    parser.add_argument("--url", required=True, help="Audio URL (e.g., http://<phone-ip>:8080/audio.wav)")
    parser.add_argument("--model", default="./models/vosk-model-fa-0.42", help="Path to Vosk model directory")
    parser.add_argument("--nogui", action="store_true", help="Run in console mode (no Tkinter GUI)")
    args = parser.parse_args()

    recognizer = load_vosk(args.model)
    ffmpeg_cmd = build_ffmpeg_cmd(args.url)

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install it (Ubuntu: sudo apt install ffmpeg).", file=sys.stderr)
        sys.exit(1)

    out_q = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=reader_thread, args=(proc, recognizer, out_q, stop_event), daemon=True)
    t.start()

    try:
        used_gui = False
        if not args.nogui:
            used_gui = run_gui_loop(out_q, stop_event)
        if not used_gui:
            console_loop(out_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
