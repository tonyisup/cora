#!/usr/bin/env python3
"""Record remaining TTS voices (resume after interruption)."""

import sys, time, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, write_pcm16_wav

try:
    import sounddevice as sd
except ImportError:
    sd = None

HP100 = butter(4, 100, btype='high', fs=SR, output='sos')

VOICES = ["Alex", "Fred", "Victoria", "Ava", "Tessa"]  # Remaining voices
PHRASES = [
    "beat bet hot caught boot",
    "boot caught hot bet beat",
    "beat hot bet boot caught",
]

def say_and_capture(text, voice, input_device=1, output_device=3):
    import subprocess, tempfile, os
    
    with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
        aiff_path = f.name
    
    try:
        subprocess.run(["say", "-v", voice, "-o", aiff_path, text], check=True, timeout=10)
        wav_path = aiff_path.replace('.aiff', '.wav')
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", aiff_path, wav_path], check=True, timeout=10)
        
        import wave
        with wave.open(wav_path, 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        
        print(f"  Playing...", end="", flush=True)
        try:
            captured = sd.playrec(audio, samplerate=SR, channels=1, dtype="float32",
                                  device=(input_device, output_device), blocking=True)
        except TypeError:
            sd.default.device = (input_device, output_device)
            captured = sd.playrec(audio, samplerate=SR, channels=1, dtype="float32", blocking=True)
        
        captured = np.asarray(captured, dtype=np.float32).reshape(-1)
        captured = sosfilt(HP100, captured).astype(np.float32)
        print(f" {len(captured)/SR:.2f}s")
        return captured
    finally:
        for p in [aiff_path, aiff_path.replace('.aiff', '.wav')]:
            try: os.unlink(p)
            except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-device', type=int, default=1)
    parser.add_argument('--output-device', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()

    output_dir = Path('data/m9_tts_captures')
    output_dir.mkdir(parents=True, exist_ok=True)

    for voice in VOICES:
        print(f"\n=== Voice: {voice} ===")
        for phrase in PHRASES:
            for rep in range(args.repeats):
                print(f"  '{phrase}' rep {rep+1}/{args.repeats}...", end="", flush=True)
                try:
                    captured = say_and_capture(phrase, voice, args.input_device, args.output_device)
                    safe_voice = voice.replace(' ', '_')
                    safe_phrase = phrase.replace(' ', '_')[:20]
                    out_path = output_dir / f"{safe_voice}_{safe_phrase}_{rep:02d}.wav"
                    write_pcm16_wav(str(out_path), captured, sr=SR)
                    print(f" -> {out_path.name}")
                except Exception as e:
                    print(f" ERROR: {e}")
                time.sleep(0.3)

    print(f"\nDone. Total captures: {len(list(output_dir.glob('*.wav')))}")

if __name__ == '__main__':
    main()
