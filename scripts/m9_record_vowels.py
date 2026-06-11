#!/usr/bin/env python3
"""Record real vowel captures through speaker->mic path for retraining."""

import sys, json, time, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, FRAME_SIZE, write_pcm16_wav
from m9_sound.vowel_transitions import DEFAULT_LOOKUP_CKPT

try:
    import sounddevice as sd
except ImportError:
    sd = None

lookup = json.loads(Path(DEFAULT_LOOKUP_CKPT).read_text())
vowel_names = lookup['vowel_names']
templates = {v: lookup['lookup'][v]['template'] for v in vowel_names}

def synth_vowel(template, duration=1.0, f0=120.0):
    n_frames = int(round(duration * SR / FRAME_SIZE))
    frames = []
    phases = {'f1': 0.0, 'f2': 0.0, 'f3': 0.0}
    for _ in range(n_frames):
        frame_t = np.arange(FRAME_SIZE, dtype=np.float32) / SR
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            frame += template[ak] * np.sin(2*np.pi*template[fk]*frame_t + phases[fk])
            phases[fk] = (phases[fk] + 2*np.pi*template[fk]*(FRAME_SIZE/SR)) % (2*np.pi)
        peak = np.max(np.abs(frame))
        if peak > 1.0:
            frame /= peak
        frames.append(frame.astype(np.float32))
    return np.concatenate(frames)

def play_and_capture(duration=1.0, input_device=1, output_device=3):
    """Play silence + vowel through speaker and capture from mic."""
    if sd is None:
        raise RuntimeError("sounddevice required")
    
    # Small lead-in silence for sync
    lead_in = np.zeros(int(0.1 * SR), dtype=np.float32)
    
    try:
        captured = sd.playrec(
            lead_in, samplerate=SR, channels=1, dtype="float32",
            device=(input_device, output_device), blocking=True,
        )
    except TypeError:
        sd.default.device = (input_device, output_device)
        captured = sd.playrec(
            lead_in, samplerate=SR, channels=1, dtype="float32", blocking=True,
        )
    
    return np.asarray(captured, dtype=np.float32).reshape(-1)

def main():
    parser = argparse.ArgumentParser(description='Record real vowel captures')
    parser.add_argument('--input-device', type=int, default=1)
    parser.add_argument('--output-device', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--duration', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path('data/m9_real_vowels')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recording real vowel captures: input={args.input_device}, output={args.output_device}")
    print(f"Repeats: {args.repeats}, Duration: {args.duration}s")
    print(f"Output: {output_dir}/\n")

    for vowel in vowel_names:
        print(f"\n=== Vowel: {vowel} ===")
        t = templates[vowel]
        audio = synth_vowel(t, duration=args.duration)
        
        captures = []
        for rep in range(args.repeats):
            print(f"  Rep {rep+1}/{args.repeats}...", end="", flush=True)
            
            # Play the vowel and capture
            try:
                captured = sd.playrec(
                    audio, samplerate=SR, channels=1, dtype="float32",
                    device=(args.input_device, args.output_device), blocking=True,
                )
            except TypeError:
                sd.default.device = (args.input_device, args.output_device)
                captured = sd.playrec(
                    audio, samplerate=SR, channels=1, dtype="float32", blocking=True,
                )
            
            captured = np.asarray(captured, dtype=np.float32).reshape(-1)
            
            # HP100 to match harness preprocessing
            hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')
            captured = sosfilt(hp_sos, captured).astype(np.float32)
            
            captures.append(captured)
            print(f" {len(captured)/SR:.2f}s")
            time.sleep(0.3)
        
        for i, cap in enumerate(captures):
            out_path = output_dir / f"{vowel}_{i:02d}.wav"
            write_pcm16_wav(str(out_path), cap, sr=SR)
            print(f"  Saved: {out_path}")

    print(f"\nDone. {len(vowel_names) * args.repeats} captures saved to {output_dir}/")

if __name__ == '__main__':
    main()
