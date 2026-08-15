# rms_meter.py
import soundcard as sc
import numpy as np

with sc.get_microphone(id="Lautsprecher (Sound BlasterX G6)", include_loopback=True).recorder(samplerate=16000, channels=1) as mic:
    while True:
        audio = mic.record(numframes=16000)  # 1 Sekunde
        rms = np.sqrt(np.mean(audio**2))
        print(f"RMS: {rms:.6f}", flush=True)