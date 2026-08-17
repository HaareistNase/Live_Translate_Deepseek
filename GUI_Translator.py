#!/usr/bin/env python3
"""
GPU Live-Transkription/Übersetzung mit GUI (tkinter) – Dark Mode.
Nimmt Loopback-Audio auf und zeigt Text mit Zeitstempel in einem dunklen Fenster an.
"""

import queue
import sys
import threading
import time
import warnings
from datetime import timedelta

import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox

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

class TranscriberApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args

        # Farben für Dark Mode
        self.bg_color = "#1e1e1e"          # dunkler Hintergrund
        self.fg_color = "#ffffff"          # weißer Text
        self.accent_color = "#4da6ff"      # helles Blau für Zeitstempel
        self.status_bg = "#2d2d2d"         # etwas hellerer Hintergrund für Statusleiste
        self.button_bg = "#3c3c3c"         # Button-Hintergrund
        self.button_fg = "#ffffff"         # Button-Text
        self.text_bg = "#000000"           # reines Schwarz für Textbereich

        self.root.title("Live Transkription / Übersetzung")
        self.root.geometry("900x600")
        self.root.configure(bg=self.bg_color)

        # Textbereich mit weichem Zeilenumbruch und dunklem Design
        self.text_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg=self.text_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,   # Cursorfarbe
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tags für Zeitstempel und Text
        self.text_area.tag_configure("timestamp", foreground=self.accent_color, font=("Segoe UI", 10, "bold"))
        self.text_area.tag_configure("text", foreground=self.fg_color, font=("Segoe UI", 12))

        # Statusleiste
        self.status_var = tk.StringVar(value="Bereit")
        status_label = tk.Label(
            root,
            textvariable=self.status_var,
            bg=self.status_bg,
            fg=self.fg_color,
            anchor=tk.W,
            padx=5,
            pady=2
        )
        status_label.pack(fill=tk.X, side=tk.BOTTOM)

        # Buttons
        button_frame = tk.Frame(root, bg=self.bg_color)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_button = tk.Button(
            button_frame,
            text="Start",
            command=self.start,
            width=10,
            bg=self.button_bg,
            fg=self.button_fg,
            activebackground=self.accent_color,
            activeforeground=self.button_fg,
            relief=tk.FLAT
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            button_frame,
            text="Stop",
            command=self.stop,
            state=tk.DISABLED,
            width=10,
            bg=self.button_bg,
            fg=self.button_fg,
            activebackground=self.accent_color,
            activeforeground=self.button_fg,
            relief=tk.FLAT
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Queue für Nachrichten vom Arbeits-Thread
        self.message_queue = queue.Queue()
        self.worker_thread = None
        self.running = False

        self.root.after(100, self.process_queue)

    def process_queue(self):
        try:
            while True:
                msg_type, content = self.message_queue.get_nowait()
                if msg_type == "text":
                    timestamp, text = content
                    self.text_area.insert(tk.END, f"[{timestamp}] ", "timestamp")
                    self.text_area.insert(tk.END, text + "\n", "text")
                    self.text_area.see(tk.END)
                elif msg_type == "status":
                    self.status_var.set(content)
                elif msg_type == "error":
                    messagebox.showerror("Fehler", content)
                    self.stop()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Starte...")

        self.worker_thread = threading.Thread(target=self.transcribe_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Gestoppt")

    def transcribe_loop(self):
        try:
            self.message_queue.put(("status", f"Lade Modell '{self.args.model}'..."))
            model = WhisperModel(self.args.model, device="cuda", compute_type="float16")

            if sc is None:
                raise ImportError("soundcard nicht installiert. Bitte: pip install soundcard")
            speakers = sc.all_speakers()
            if self.args.device is None:
                speaker = sc.default_speaker()
            else:
                try:
                    idx = int(self.args.device)
                    speaker = speakers[idx]
                except (ValueError, IndexError):
                    speaker = None
                    for sp in speakers:
                        if self.args.device.lower() in sp.name.lower():
                            speaker = sp
                            break
                    if speaker is None:
                        raise ValueError(f"Lautsprecher '{self.args.device}' nicht gefunden.")

            self.message_queue.put(("status", f"Aufnahme läuft ({speaker.name})"))

            block_size = int(self.args.sample_rate * self.args.chunk_size)
            start_time = time.time()

            with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(
                    samplerate=self.args.sample_rate, channels=1) as mic:
                while self.running:
                    audio = mic.record(numframes=block_size)
                    if audio.ndim > 1:
                        audio = audio.flatten()

                    if self.is_silent(audio, self.args.silence_threshold):
                        continue

                    segments, info = model.transcribe(
                        audio,
                        language=self.args.language,
                        task="translate" if self.args.translate else "transcribe",
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False,
                        temperature=0.0,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        initial_prompt="Transcribe all words literally, including profanity and slang.",
                    )

                    text = " ".join([seg.text for seg in segments]).strip()
                    if text and not self.is_hallucination(text):
                        elapsed = timedelta(seconds=int(time.time() - start_time))
                        self.message_queue.put(("text", (str(elapsed), text)))

        except Exception as e:
            self.message_queue.put(("error", str(e)))
        finally:
            self.running = False
            self.message_queue.put(("status", "Beendet"))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

    @staticmethod
    def is_silent(audio, threshold):
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold

    @staticmethod
    def is_hallucination(text):
        lower = text.lower().strip()
        return any(phrase in lower for phrase in HALLUCINATION_PHRASES)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="Index oder Name des Lautsprechers (z. B. 4)")
    parser.add_argument("--model", default="medium", help="Whisper-Modell (z. B. small, medium, large-v3)")
    parser.add_argument("--chunk-size", type=int, default=15, help="Chunk-Dauer in Sekunden")
    parser.add_argument("--language", default=None, help="Sprachcode (z. B. de, en)")
    parser.add_argument("--translate", action="store_true", help="Ins Englische übersetzen")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--silence-threshold", type=float, default=0.0001,
                        help="RMS-Schwelle für Stille (Standard: 0.0001)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = TranscriberApp(root, args)
    root.mainloop()