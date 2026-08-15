#!/usr/bin/env python3
"""
Transkribiert oder übersetzt das Audio, das von einer Soundkarte (z. B. Soundblaster G6)
ausgegeben wird. Unterstützt zwei Modi:
  1. Normales Aufnahmegerät (z. B. "What U Hear", Stereo Mix) über sounddevice.
  2. WASAPI-Loopback (nimmt direkt den Lautsprecher-Output auf) über soundcard.

Verwendung:
    python soundcard_transcriber.py --list-devices
    python soundcard_transcriber.py --device "What U Hear (Sound Blaster G6)"
    python soundcard_transcriber.py --loopback [--device "Lautsprecher (Sound BlasterX G6)"]

Optionen:
    --list-devices  Zeigt alle verfügbaren Audio-Geräte an und beendet.
    --device        Name oder Index des Aufnahmegeräts. Bei --loopback: Name/Index des AUSGABEgeräts.
    --loopback      Aktiviert WASAPI-Loopback (Windows). Nimmt den Ton des angegebenen Ausgabegeräts auf.
                    Ohne --device wird das Standard-Ausgabegerät verwendet.
    --model         Whisper-Modell (tiny, base, small, medium, large) [Standard: base]
    --chunk-size    Dauer eines Verarbeitungsblocks in Sekunden [Standard: 10]
    --language      Sprache des Audios (z. B. de, en, fr). Automatisch, wenn nicht gesetzt.
    --translate     Übersetzt die Transkription direkt ins Englische.
    --sample-rate   Abtastrate für Whisper [Standard: 16000]
"""

import argparse
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import whisper

# Für normale Aufnahme
import sounddevice as sd

# Für Loopback-Aufnahme (Windows)
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
except ImportError:
    SOUNDCARD_AVAILABLE = False


# Warnungen unterdrücken
# 1. FP16-Warnung von Whisper (nur auf CPU relevant)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# 2. Soundcard-Warnung bei Datenlücken
warnings.filterwarnings("ignore", message="data discontinuity in recording")


def list_audio_devices():
    """Gibt alle verfügbaren Audio-Geräte aus (sounddevice)."""
    print("=== Verfügbare Audio-Geräte (sounddevice) ===")
    print(sd.query_devices())
    print("\nHinweise:")
    print("- Normale Aufnahmegeräte: 'What U Hear', 'Stereo Mix', Mikrofon, etc.")
    print("- Für Loopback (Lautsprecher-Output) wird die Bibliothek 'soundcard' verwendet.")
    print("  Führe 'pip install soundcard' aus, falls nicht installiert.")
    print("  Die Geräte können über '--list-devices-loopback' angezeigt werden.")


def list_loopback_devices():
    """Listet verfügbare Ausgabegeräte für Loopback auf (soundcard)."""
    if not SOUNDCARD_AVAILABLE:
        print("soundcard ist nicht installiert. Bitte installiere es mit: pip install soundcard")
        sys.exit(1)

    print("=== Verfügbare Ausgabegeräte (Lautsprecher) für Loopback ===")
    speakers = sc.all_speakers()
    for i, sp in enumerate(speakers):
        print(f"{i}: {sp.name}")
    print("\nVerwende den Index oder einen Teil des Namens mit --device.")


def find_speaker_by_name_or_index(device_spec):
    """Findet ein Ausgabegerät (Speaker) anhand Index oder Name (Teilübereinstimmung)."""
    speakers = sc.all_speakers()
    if device_spec is None:
        return sc.default_speaker()
    try:
        idx = int(device_spec)
        if 0 <= idx < len(speakers):
            return speakers[idx]
        else:
            print(f"Fehler: Index {idx} außerhalb des Bereichs.")
            sys.exit(1)
    except ValueError:
        search_name = device_spec.lower()
        for sp in speakers:
            if search_name in sp.name.lower():
                return sp
        print(f"Fehler: Lautsprecher '{device_spec}' nicht gefunden.")
        print("Verfügbare Lautsprecher:")
        for i, sp in enumerate(speakers):
            print(f"{i}: {sp.name}")
        sys.exit(1)


def get_device_index_sounddevice(device_spec, kind='input'):
    """Ermittelt den Geräteindex für sounddevice (normale Aufnahme)."""
    if device_spec is None:
        return None
    try:
        idx = int(device_spec)
        sd.query_devices(idx)
        return idx
    except (ValueError, IndexError):
        devices = sd.query_devices()
        search_name = device_spec.lower()
        for i, dev in enumerate(devices):
            if search_name in dev['name'].lower():
                if kind == 'input' and dev['max_input_channels'] > 0:
                    return i
                elif kind == 'output' and dev['max_output_channels'] > 0:
                    return i
                elif kind is None:
                    return i
        print(f"Fehler: Gerät '{device_spec}' nicht gefunden.")
        print("Verwende --list-devices, um alle Geräte anzuzeigen.")
        sys.exit(1)


def audio_callback_sounddevice(indata, frames, time_info, status, audio_queue):
    """Callback für sounddevice.Stream."""
    if status:
        print(f"Stream-Status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def stream_audio_chunks_sounddevice(device_index, chunk_size_sec, sample_rate):
    """Generator für normale Aufnahme über sounddevice."""
    import queue

    q = queue.Queue(maxsize=10)
    block_size = int(sample_rate * chunk_size_sec)

    with sd.InputStream(device=device_index, channels=1, samplerate=sample_rate,
                        blocksize=block_size, dtype='float32',
                        callback=lambda indata, frames, t, status: audio_callback_sounddevice(indata, frames, t, status, q)):
        print(f"Aufnahme gestartet (sounddevice, Gerät: {device_index}, Chunk: {chunk_size_sec}s, Abtastrate: {sample_rate} Hz)")
        while True:
            data = q.get()
            yield data


def stream_audio_chunks_loopback(speaker, chunk_size_sec, sample_rate):
    """Generator für Loopback-Aufnahme über soundcard."""
    block_size = int(sample_rate * chunk_size_sec)

    with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(samplerate=sample_rate, channels=1) as mic:
        print(f"Loopback-Aufnahme gestartet (Lautsprecher: {speaker.name}, Chunk: {chunk_size_sec}s, Abtastrate: {sample_rate} Hz)")
        while True:
            data = mic.record(numframes=block_size)
            # soundcard gibt bereits float32 zurück, wir stellen sicher, dass es eindimensional ist
            if data.ndim > 1:
                data = data.flatten()
            yield data


def main():
    parser = argparse.ArgumentParser(description="Audio von Soundkarte (Loopback) mit Whisper transkribieren")
    parser.add_argument("--list-devices", action="store_true", help="Audio-Geräte (sounddevice) auflisten und beenden")
    parser.add_argument("--list-devices-loopback", action="store_true", help="Ausgabegeräte für Loopback auflisten und beenden")
    parser.add_argument("--device", default=None, help="Name oder Index des Geräts (abhängig von Modus)")
    parser.add_argument("--loopback", action="store_true", help="WASAPI-Loopback für Ausgabegerät aktivieren (Windows)")
    parser.add_argument("--model", default="base", help="Whisper-Modell (tiny, base, small, medium, large)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Dauer eines Chunks in Sekunden")
    parser.add_argument("--language", default=None, help="Sprachcode (z. B. de, en, fr)")
    parser.add_argument("--translate", action="store_true", help="Direkt ins Englische übersetzen")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Abtastrate (Standard 16000 Hz)")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return
    if args.list_devices_loopback:
        list_loopback_devices()
        return

    # Whisper-Modell laden
    print(f"Lade Whisper-Modell '{args.model}' ...")
    model = whisper.load_model(args.model)

    start_time = time.time()
    try:
        if args.loopback:
            if not SOUNDCARD_AVAILABLE:
                print("Fehler: 'soundcard' ist nicht installiert. Bitte installiere es mit: pip install soundcard")
                sys.exit(1)
            speaker = find_speaker_by_name_or_index(args.device)
            audio_generator = stream_audio_chunks_loopback(speaker, args.chunk_size, args.sample_rate)
        else:
            device_idx = get_device_index_sounddevice(args.device, kind='input')
            audio_generator = stream_audio_chunks_sounddevice(device_idx, args.chunk_size, args.sample_rate)

        for audio_chunk in audio_generator:
            options = {}
            if args.language:
                options["language"] = args.language
            if args.translate:
                options["task"] = "translate"

            result = model.transcribe(audio_chunk, **options)
            text = result["text"].strip()
            if text:
                elapsed = timedelta(seconds=int(time.time() - start_time))
                print(f"[{elapsed}] {text}", flush=True)

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    except Exception as e:
        print(f"Fehler: {e}", file=sys.stderr)

    print("Aufnahme beendet.")


if __name__ == "__main__":
    main()