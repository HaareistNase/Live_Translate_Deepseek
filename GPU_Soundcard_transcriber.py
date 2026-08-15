#!/usr/bin/env python3
"""
GPU Live-Transkription mit einfacher Anti-Halluzination.
Nur Stille-Filter + Phrasen-Blacklist.
"""

import argparse
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
from faster_whisper import WhisperModel

try:
    import soundcard as sc
except ImportError:
    sc = None

warnings.filterwarnings("ignore")

HALLUCINATION_PHRASES = [
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe to my channel",
    "don't forget to subscribe",
    "see you in the next video",
    "thank you for watching",
    "please like and subscribe",
    "comment below",
]

def list_loopback_devices():
    if sc is None:
        print("soundcard fehlt. Bitte: pip install soundcard")
        sys.exit(1)
    print("=== Lautsprecher ===")
    for i, sp in enumerate(sc.all_speakers()):
        print(f"{i}: {sp.name}")

def find_speaker(spec):
    speakers = sc.all_speakers()
    if spec is None:
        return sc.default_speaker()
    try:
        idx = int(spec)
        return speakers[idx]
    except (ValueError, IndexError):
        for sp in speakers:
            if spec.lower() in sp.name.lower():
                return sp
        print(f"Lautsprecher '{spec}' nicht gefunden.")
        sys.exit(1)

def is_silent(audio, threshold):
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold

def is_hallucination(text):
    lower = text.lower().strip()
    return any(phrase in lower for phrase in HALLUCINATION_PHRASES)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--language", default=None)
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--silence-threshold", type=float, default=0.005,
                        help="RMS-Schwelle für Stille (Standard: 0.005). Passe an deine Messung an.")
    args = parser.parse_args()

    if args.list_devices:
        list_loopback_devices()
        return

    print(f"Lade faster-whisper Modell '{args.model}' auf GPU ...")
    model = WhisperModel(args.model, device="cuda", compute_type="float16")

    speaker = find_speaker(args.device)
    block_size = int(args.sample_rate * args.chunk_size)

    print(f"Starte Loopback-Aufnahme (Lautsprecher: {speaker.name}, Chunk: {args.chunk_size}s)")
    print(f"Stille-Schwelle: {args.silence_threshold:.4f}")
    start_time = time.time()

    with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(
            samplerate=args.sample_rate, channels=1) as mic:
        while True:
            try:
                audio = mic.record(numframes=block_size)
                if audio.ndim > 1:
                    audio = audio.flatten()

                # Nur verarbeiten, wenn genug Energie (Sprache) vorhanden
                if is_silent(audio, args.silence_threshold):
                    continue

                segments, info = model.transcribe(
                    audio,
                    language=args.language,
                    task="translate" if args.translate else "transcribe",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False,
                    temperature=0.0,
		    repetition_penalty=1.2,
		    no_repeat_ngram_size=3,
                )

                text = " ".join([seg.text for seg in segments]).strip()
                if text and not is_hallucination(text):
                    elapsed = timedelta(seconds=int(time.time() - start_time))
                    print(f"[{elapsed}] {text}", flush=True)

            except KeyboardInterrupt:
                print("\nAbbruch durch Benutzer.")
                break
            except Exception as e:
                print(f"Fehler: {e}", file=sys.stderr)
                time.sleep(1)

    print("Aufnahme beendet.")

if __name__ == "__main__":
    main()