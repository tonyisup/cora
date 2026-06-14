#!/usr/bin/env python3
"""M0 consonant corpus capture — interactive with countdown.

Records consonant-bearing syllables with a spoken prompt and countdown.
Designed for real human speech capture.

Usage:
    ./.venv/bin/python scripts/m10_record_human.py --speaker juicebox
"""

import sys, json, time, uuid, wave, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import write_pcm16_wav

DEFAULT_SR = 48000

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice required")
    sys.exit(1)

PROMPTS = [
    # Stops
    {"text": "pa", "label": "p", "category": "stop"},
    {"text": "ta", "label": "t", "category": "stop"},
    {"text": "ka", "label": "k", "category": "stop"},
    {"text": "ba", "label": "b", "category": "stop"},
    {"text": "da", "label": "d", "category": "stop"},
    {"text": "ga", "label": "g", "category": "stop"},
    # Fricatives
    {"text": "sa", "label": "s", "category": "fricative"},
    {"text": "fa", "label": "f", "category": "fricative"},
    {"text": "sha", "label": "sh", "category": "fricative"},
    {"text": "va", "label": "v", "category": "fricative"},
    {"text": "za", "label": "z", "category": "fricative"},
    {"text": "tha", "label": "th", "category": "fricative"},
    # Nasals
    {"text": "ma", "label": "m", "category": "nasal"},
    {"text": "na", "label": "n", "category": "nasal"},
    # Liquids/Glides
    {"text": "la", "label": "l", "category": "liquid"},
    {"text": "ra", "label": "r", "category": "liquid"},
    {"text": "wa", "label": "w", "category": "glide"},
    {"text": "ya", "label": "y", "category": "glide"},
    # VCV patterns
    {"text": "apa", "label": "p", "category": "stop"},
    {"text": "ata", "label": "t", "category": "stop"},
    {"text": "aka", "label": "k", "category": "stop"},
    {"text": "asa", "label": "s", "category": "fricative"},
    {"text": "afa", "label": "f", "category": "fricative"},
    {"text": "ama", "label": "m", "category": "nasal"},
    {"text": "ana", "label": "n", "category": "nasal"},
    # CVCV
    {"text": "pata", "label": "p", "category": "stop"},
    {"text": "safa", "label": "s", "category": "fricative"},
    {"text": "mana", "label": "m", "category": "nasal"},
    {"text": "lara", "label": "l", "category": "liquid"},
]

def play_tone(freq=440, duration=0.15, sr=DEFAULT_SR, output_device=3):
    """Play a short beep as recording cue."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(tone, samplerate=sr, device=output_device)
    sd.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default="juicebox")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--countdown", type=float, default=1.5)
    args = parser.parse_args()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/m10/raw") / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/m10/manifests") / "captures.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS)
    print("=" * 78)
    print("M10 Consonant Capture — Human Speech")
    print("=" * 78)
    print(f"  Prompts:    {total}")
    print(f"  Speaker:    {args.speaker}")
    print(f"  Rate:       {args.rate} Hz")
    print(f"  Duration:   {args.duration}s per prompt")
    print(f"  Countdown:  {args.countdown}s before each recording")
    print(f"  Output:     {output_dir}")
    print()
    print("  A beep will sound before each recording.")
    print("  Speak the prompt clearly into the mic after the beep.")
    print("  Press Ctrl+C to stop at any time.")
    print()

    rows = []
    try:
        for idx, p in enumerate(PROMPTS):
            prompt_text = p["text"]
            label = p["label"]
            category = p["category"]

            print(f"[{idx+1}/{total}] '{prompt_text}' ({label}, {category})")
            print(f"  Get ready...", end="", flush=True)
            time.sleep(args.countdown * 0.5)
            print(f" 3...", end="", flush=True)
            time.sleep(args.countdown * 0.25)
            print(f" 2...", end="", flush=True)
            time.sleep(args.countdown * 0.25)
            print(f" 1...", end="", flush=True)
            time.sleep(args.countdown * 0.25)
            print(f" NOW!", flush=True)

            # Play beep cue
            play_tone(880, 0.1, sr=args.rate, output_device=args.output_device)
            time.sleep(0.05)

            # Record
            n_samples = int(round(args.rate * args.duration))
            recorded = sd.rec(n_samples, samplerate=args.rate, channels=1,
                              device=args.input_device, dtype="float32")
            sd.wait()
            audio = np.asarray(recorded, dtype=np.float32).reshape(-1)

            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.max(np.abs(audio)))

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            capture_id = f"hu_{str(uuid.uuid4())[:8]}"
            filename = f"{timestamp}_{label}_{capture_id}.wav"
            wav_path = output_dir / filename
            write_pcm16_wav(wav_path, audio, sr=args.rate)

            row = {
                "capture_id": capture_id,
                "session_id": session_id,
                "speaker": args.speaker,
                "prompt": prompt_text,
                "label": label,
                "category": category,
                "wav_path": str(wav_path.relative_to(Path("."))),
                "wav_path_48k": str(wav_path.relative_to(Path("."))),
                "sample_rate": args.rate,
                "duration_s": round(len(audio) / args.rate, 3),
                "rms": rms,
                "peak": peak,
                "source": "human",
                "model_version": "m10_48k",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            quality = "GOOD" if rms > 0.03 else "LOW" if rms > 0.01 else "QUIET"
            print(f"  Saved: {filename}  rms={rms:.4f} peak={peak:.4f} [{quality}]")
            print()

            # Short pause between prompts
            time.sleep(0.3)

    except KeyboardInterrupt:
        print(f"\nStopped by user.")

    print(f"\nSession complete: {len(rows)}/{total} captures saved")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")

    if rows:
        by_label = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(r["rms"])
        print("\nPer-label summary:")
        for label in sorted(by_label):
            rms_list = by_label[label]
            print(f"  {label:4s}: {len(rms_list)} captures, avg rms={np.mean(rms_list):.4f}")


if __name__ == "__main__":
    main()
