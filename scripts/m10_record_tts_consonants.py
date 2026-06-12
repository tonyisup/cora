#!/usr/bin/env python3
"""M10 TTS consonant capture.

Uses macOS `say` to generate consonant-vowel prompts through the speaker,
captures them via the mic, and saves them as M10 training data.

Usage:
    ./.venv/bin/python scripts/m10_record_tts_consonants.py
    ./.venv/bin/python scripts/m10_record_tts_consonants.py --voices Samantha Daniel
"""

import sys, json, time, uuid, subprocess, tempfile, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, write_pcm16_wav

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice required: pip install sounddevice")
    sys.exit(1)

HP100 = butter(4, 100, btype='high', fs=SR, output='sos')

# Consonant prompt definitions
PROMPTS = [
    # Single CV syllables
    {"text": "pa", "label": "p", "category": "stop"},
    {"text": "ta", "label": "t", "category": "stop"},
    {"text": "ka", "label": "k", "category": "stop"},
    {"text": "sa", "label": "s", "category": "fricative"},
    {"text": "fa", "label": "f", "category": "fricative"},
    {"text": "sha", "label": "sh", "category": "fricative"},
    {"text": "ma", "label": "m", "category": "nasal"},
    {"text": "na", "label": "n", "category": "nasal"},
    # CVCV sequences
    {"text": "pa ta", "label": "p", "category": "stop"},
    {"text": "ta ka", "label": "t", "category": "stop"},
    {"text": "ka pa", "label": "k", "category": "stop"},
    {"text": "sa fa", "label": "s", "category": "fricative"},
    {"text": "fa sha", "label": "f", "category": "fricative"},
    {"text": "sha sa", "label": "sh", "category": "fricative"},
    {"text": "ma na", "label": "m", "category": "nasal"},
    {"text": "na ma", "label": "n", "category": "nasal"},
    # Vowel-consonant-vowel
    {"text": "apa", "label": "p", "category": "stop"},
    {"text": "ata", "label": "t", "category": "stop"},
    {"text": "aka", "label": "k", "category": "stop"},
    {"text": "asa", "label": "s", "category": "fricative"},
    {"text": "afa", "label": "f", "category": "fricative"},
    {"text": "asha", "label": "sh", "category": "fricative"},
    {"text": "ama", "label": "m", "category": "nasal"},
    {"text": "ana", "label": "n", "category": "nasal"},
]

VOICES = ["Samantha", "Daniel", "Alex", "Victoria", "Tessa"]


def tts_to_wav(text: str, voice: str, out_wav: Path) -> bool:
    """Generate TTS audio using macOS `say` and convert to WAV."""
    with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
        aiff_path = f.name
    try:
        subprocess.run(["say", "-v", voice, "-o", aiff_path, text], check=True, timeout=15)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", aiff_path, str(out_wav)],
                       check=True, timeout=10)
        return True
    except Exception as e:
        print(f"  TTS error: {e}")
        return False
    finally:
        Path(aiff_path).unlink(missing_ok=True)


def play_and_capture(audio: np.ndarray, input_device: int, output_device: int) -> np.ndarray:
    """Play audio through speaker and record from mic simultaneously."""
    try:
        captured = sd.playrec(audio, samplerate=SR, channels=1, dtype="float32",
                              device=(input_device, output_device), blocking=True)
    except TypeError:
        sd.default.device = (input_device, output_device)
        captured = sd.playrec(audio, samplerate=SR, channels=1, dtype="float32", blocking=True)
    captured = np.asarray(captured, dtype=np.float32).reshape(-1)
    return sosfilt(HP100, captured).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 TTS consonant capture")
    parser.add_argument("--voices", nargs="+", default=VOICES, help="TTS voices to use")
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=3)
    parser.add_argument("--speaker", type=str, default="tts", help="Speaker prefix for manifest")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without recording")
    args = parser.parse_args()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/m10/raw") / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/m10/manifests") / "captures.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(args.voices)
    print("=" * 78)
    print("M10 TTS Consonant Capture")
    print("=" * 78)
    print(f"  Prompts:     {len(PROMPTS)}")
    print(f"  Voices:      {', '.join(args.voices)} ({len(args.voices)})")
    print(f"  Total:       {total} captures")
    print(f"  Input dev:   {args.input_device}")
    print(f"  Output dev:  {args.output_device}")
    print(f"  Output dir:  {output_dir}")
    print()

    if args.dry_run:
        for voice in args.voices:
            for p in PROMPTS:
                print(f"  [{voice:10s}] {p['text']:15s} -> {p['label']} ({p['category']})")
        return

    rows = []
    count = 0

    for voice in args.voices:
        print(f"\n--- Voice: {voice} ---")
        for p in PROMPTS:
            count += 1
            text = p["text"]
            label = p["label"]

            print(f"[{count}/{total}] {voice}: '{text}' -> {label}", end="", flush=True)

            # Generate TTS
            tts_wav = output_dir / f"tts_{voice}_{label}_{count:03d}.wav"
            if not tts_to_wav(text, voice, tts_wav):
                print(" [TTS FAIL]")
                continue

            # Load TTS audio
            import wave
            with wave.open(str(tts_wav), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

            # Play and capture
            captured = play_and_capture(audio, args.input_device, args.output_device)

            rms = float(np.sqrt(np.mean(captured**2)))
            peak = float(np.max(np.abs(captured)))

            # Save captured audio
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            capture_id = f"{voice[:3].lower()}_{str(uuid.uuid4())[:6]}"
            filename = f"{timestamp}_{label}_{capture_id}.wav"
            wav_path = output_dir / filename
            write_pcm16_wav(wav_path, captured, sr=SR)

            row = {
                "capture_id": capture_id,
                "session_id": session_id,
                "speaker": f"{args.speaker}_{voice}",
                "prompt": text,
                "label": label,
                "category": p["category"],
                "wav_path": str(wav_path.relative_to(Path("."))),
                "sample_rate": SR,
                "duration_s": round(len(captured) / SR, 3),
                "rms": rms,
                "peak": peak,
                "input_device": args.input_device,
                "output_device": args.output_device,
                "voice": voice,
                "tts_wav": str(tts_wav.relative_to(Path("."))),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            quality = "GOOD" if rms > 0.02 else "LOW" if rms > 0.008 else "SILENT"
            print(f"  rms={rms:.4f} peak={peak:.4f} [{quality}]")

    print(f"\nSession complete: {len(rows)} captures saved")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Summary
    by_label = {}
    for r in rows:
        l = r["label"]
        if l not in by_label:
            by_label[l] = {"count": 0, "rms_list": []}
        by_label[l]["count"] += 1
        by_label[l]["rms_list"].append(r["rms"])

    print("\nPer-label summary:")
    for label in sorted(by_label.keys()):
        info = by_label[label]
        avg_rms = float(np.mean(info["rms_list"]))
        print(f"  {label:4s}: {info['count']:3d} captures, avg rms={avg_rms:.4f}")


if __name__ == "__main__":
    main()
