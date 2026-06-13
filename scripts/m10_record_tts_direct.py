#!/usr/bin/env python3
"""M10 TTS consonant capture — direct (no speaker→mic).

Generates TTS audio using macOS `say` and saves it directly as training data
without going through the speaker→mic loop. This preserves high-frequency
content (sibilants, fricatives) that is lost in the speaker→mic path.

Usage:
    ./.venv/bin/python scripts/m10_record_tts_direct.py
    scripts/m10_record_tts_direct.py --voices Samantha Daniel Alex Victoria Tessa
"""

import sys, json, subprocess, tempfile, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, write_pcm16_wav

# Consonant prompt definitions
PROMPTS = [
    # Single CV syllables
    {"text": "pa", "label": "p", "category": "stop"},
    {"text": "pi", "label": "p", "category": "stop"},
    {"text": "pu", "label": "p", "category": "stop"},
    {"text": "ta", "label": "t", "category": "stop"},
    {"text": "ti", "label": "t", "category": "stop"},
    {"text": "tu", "label": "t", "category": "stop"},
    {"text": "ka", "label": "k", "category": "stop"},
    {"text": "ki", "label": "k", "category": "stop"},
    {"text": "ku", "label": "k", "category": "stop"},
    {"text": "sa", "label": "s", "category": "fricative"},
    {"text": "si", "label": "s", "category": "fricative"},
    {"text": "su", "label": "s", "category": "fricative"},
    {"text": "fa", "label": "f", "category": "fricative"},
    {"text": "fi", "label": "f", "category": "fricative"},
    {"text": "fu", "label": "f", "category": "fricative"},
    {"text": "sha", "label": "sh", "category": "fricative"},
    {"text": "shi", "label": "sh", "category": "fricative"},
    {"text": "shu", "label": "sh", "category": "fricative"},
    {"text": "ma", "label": "m", "category": "nasal"},
    {"text": "mi", "label": "m", "category": "nasal"},
    {"text": "mu", "label": "m", "category": "nasal"},
    {"text": "na", "label": "n", "category": "nasal"},
    {"text": "ni", "label": "n", "category": "nasal"},
    {"text": "nu", "label": "n", "category": "nasal"},
    # Multi-syllable
    {"text": "pa ta ka", "label": "p", "category": "stop"},
    {"text": "sa fa sha", "label": "s", "category": "fricative"},
    {"text": "ma na pa", "label": "m", "category": "nasal"},
    {"text": "ta ka sa", "label": "t", "category": "stop"},
    {"text": "fa sha ma", "label": "f", "category": "fricative"},
    {"text": "na ma ta", "label": "n", "category": "nasal"},
    {"text": "ka pa na", "label": "k", "category": "stop"},
    {"text": "sha fa sa", "label": "sh", "category": "fricative"},
    # Additional variations for more data
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


def tts_to_audio(text: str, voice: str) -> np.ndarray:
    """Generate TTS audio and return as numpy array at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
        aiff_path = f.name
    try:
        subprocess.run(["say", "-v", voice, "-o", aiff_path, text], check=True, timeout=15)
        wav_path = aiff_path.replace('.aiff', '.wav')
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", aiff_path, wav_path],
                       check=True, timeout=10)
        
        import wave
        with wave.open(wav_path, 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        Path(aiff_path).unlink(missing_ok=True)
        Path(aiff_path.replace('.aiff', '.wav')).unlink(missing_ok=True)


def compute_hf_pct(audio: np.ndarray) -> float:
    """Compute high-frequency energy percentage."""
    n_fft = min(1024, len(audio))
    spec = np.abs(np.fft.rfft(audio[:n_fft]))
    freqs = np.linspace(0, SR / 2, len(spec))
    low_e = np.mean(spec[freqs < 4000] ** 2)
    high_e = np.mean(spec[freqs >= 4000] ** 2)
    total_e = low_e + high_e
    return high_e / total_e * 100 if total_e > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 TTS consonant capture (direct)")
    parser.add_argument("--voices", nargs="+", default=VOICES)
    parser.add_argument("--speaker", type=str, default="tts_direct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/m10/raw") / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/m10/manifests") / "captures.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(args.voices)
    print("=" * 78)
    print("M10 TTS Consonant Capture — Direct (no speaker→mic)")
    print("=" * 78)
    print(f"  Prompts:     {len(PROMPTS)}")
    print(f"  Voices:      {', '.join(args.voices)} ({len(args.voices)})")
    print(f"  Total:       {total} captures")
    print(f"  Rate:        {SR} Hz")
    print()

    if args.dry_run:
        for voice in args.voices:
            for p in PROMPTS:
                print(f"  [{voice:10s}] {p['text']:15s} -> {p['label']} ({p['category']})")
        return

    rows = []
    count = 0

    for voice in args.voices:
        print(f"\n=== Voice: {voice} ===")
        for p in PROMPTS:
            count += 1
            text = p["text"]
            label = p["label"]

            print(f"[{count}/{total}] {voice}: '{text}' -> {label}", end="", flush=True)

            try:
                audio = tts_to_audio(text, voice)
            except Exception as e:
                print(f" [TTS ERR: {e}]")
                continue

            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))
            hf_pct = compute_hf_pct(audio)

            # Save audio
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{label}_{voice[:3].lower()}.wav"
            wav_path = output_dir / filename
            write_pcm16_wav(wav_path, audio, sr=SR)

            capture_id = f"{voice[:3].lower()}_{timestamp[-6:]}"
            row = {
                "capture_id": capture_id,
                "session_id": session_id,
                "speaker": f"{args.speaker}_{voice}",
                "prompt": text,
                "label": label,
                "category": p["category"],
                "wav_path": str(wav_path.relative_to(Path("."))),
                "sample_rate": SR,
                "duration_s": round(len(audio) / SR, 3),
                "rms": rms,
                "peak": peak,
                "hf_pct": float(hf_pct),
                "voice": voice,
                "source": "tts_direct",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            quality = "GOOD" if rms > 0.05 else "LOW" if rms > 0.02 else "SILENT"
            print(f"  rms={rms:.4f} hf={hf_pct:.0f}% [{quality}]")

    print(f"\nSession complete: {len(rows)} captures saved")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Summary by label
    by_label = {}
    for r in rows:
        l = r["label"]
        if l not in by_label:
            by_label[l] = {"count": 0, "rms_list": [], "hf_list": []}
        by_label[l]["count"] += 1
        by_label[l]["rms_list"].append(r["rms"])
        by_label[l]["hf_list"].append(r.get("hf_pct", 0))

    print("\nPer-label summary:")
    for label in sorted(by_label.keys()):
        info = by_label[label]
        avg_rms = float(np.mean(info["rms_list"]))
        avg_hf = float(np.mean(info["hf_list"]))
        print(f"  {label:4s}: {info['count']:3d} captures, avg rms={avg_rms:.4f}, avg hf={avg_hf:.0f}%")


if __name__ == "__main__":
    main()
