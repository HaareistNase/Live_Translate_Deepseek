@echo off
REM Startet die Live-Transkription/Übersetzung mit GPU
REM Parameter nach Belieben anpassen

python GPU_Soundcard_transcriber.py --device 4 --model medium --chunk-size 15 --translate --silence-threshold 0.0001
