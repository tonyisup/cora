#!/usr/bin/env python3
"""M10 TTS consonant capture — large dataset generator.

Generates a large, diverse set of consonant training examples using macOS `say`.
Applies speaker→mic domain adaptation (6kHz LP filter) to direct TTS audio.

Usage:
    ./.venv/bin/python scripts/m10_record_tts_direct.py --voices Samantha Daniel Alex Victoria Tessa
"""

import sys, json, subprocess, tempfile, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.temporal import SR, write_pcm16_wav
from scipy.signal import butter, sosfilt

# Speaker→mic simulation
_SPEAKER_LP = butter(4, 6000, btype='low', fs=SR, output='sos')

# Large prompt set with many variations
PROMPTS = [
    # Single CV with all vowel contexts
    {"text": "pa", "label": "p"}, {"text": "pi", "label": "p"}, {"text": "pu", "label": "p"},
    {"text": "pe", "label": "p"}, {"text": "po", "label": "p"},
    {"text": "ta", "label": "t"}, {"text": "ti", "label": "t"}, {"text": "tu", "label": "t"},
    {"text": "te", "label": "t"}, {"text": "to", "label": "t"},
    {"text": "ka", "label": "k"}, {"text": "ki", "label": "k"}, {"text": "ku", "label": "k"},
    {"text": "ke", "label": "k"}, {"text": "ko", "label": "k"},
    {"text": "sa", "label": "s"}, {"text": "si", "label": "s"}, {"text": "su", "label": "s"},
    {"text": "se", "label": "s"}, {"text": "so", "label": "s"},
    {"text": "fa", "label": "f"}, {"text": "fi", "label": "f"}, {"text": "fu", "label": "f"},
    {"text": "fe", "label": "f"}, {"text": "fo", "label": "f"},
    {"text": "sha", "label": "sh"}, {"text": "shi", "label": "sh"}, {"text": "shu", "label": "sh"},
    {"text": "she", "label": "sh"}, {"text": "sho", "label": "sh"},
    {"text": "ma", "label": "m"}, {"text": "mi", "label": "m"}, {"text": "mu", "label": "m"},
    {"text": "me", "label": "m"}, {"text": "mo", "label": "m"},
    {"text": "na", "label": "n"}, {"text": "ni", "label": "n"}, {"text": "nu", "label": "n"},
    {"text": "ne", "label": "n"}, {"text": "no", "label": "n"},
    # VCV (vowel-consonant-vowel)
    {"text": "apa", "label": "p"}, {"text": "api", "label": "p"}, {"text": "apu", "label": "p"},
    {"text": "ata", "label": "t"}, {"text": "ati", "label": "t"}, {"text": "atu", "label": "t"},
    {"text": "aka", "label": "k"}, {"text": "aki", "label": "k"}, {"text": "aku", "label": "k"},
    {"text": "asa", "label": "s"}, {"text": "asi", "label": "s"}, {"text": "asu", "label": "s"},
    {"text": "afa", "label": "f"}, {"text": "afi", "label": "f"}, {"text": "afu", "label": "f"},
    {"text": "asha", "label": "sh"}, {"text": "ashi", "label": "sh"}, {"text": "ashu", "label": "sh"},
    {"text": "ama", "label": "m"}, {"text": "ami", "label": "m"}, {"text": "amu", "label": "m"},
    {"text": "ana", "label": "n"}, {"text": "ani", "label": "n"}, {"text": "anu", "label": "n"},
    # CVCV sequences
    {"text": "pata", "label": "p"}, {"text": "tapa", "label": "t"}, {"text": "kapa", "label": "k"},
    {"text": "safa", "label": "s"}, {"text": "fasa", "label": "f"}, {"text": "shasa", "label": "sh"},
    {"text": "mana", "label": "m"}, {"text": "nama", "label": "n"}, {"text": "pana", "label": "p"},
    {"text": "taka", "label": "t"}, {"text": "kata", "label": "k"}, {"text": "saka", "label": "s"},
    {"text": "fasha", "label": "f"}, {"text": "shafa", "label": "sh"}, {"text": "mata", "label": "m"},
    # Multi-syllable
    {"text": "pa ta ka sa", "label": "p"}, {"text": "sa fa sha ma", "label": "s"},
    {"text": "ma na pa ta", "label": "m"}, {"text": "ta ka sa fa", "label": "t"},
    {"text": "ka pa na ma", "label": "k"}, {"text": "fa sha ma na", "label": "f"},
    {"text": "na ma ta ka", "label": "n"}, {"text": "sha fa sa ma", "label": "sh"},
]

# All available English voices
ALL_VOICES = [
    "Samantha", "Daniel", "Alex", "Victoria", "Tessa",
    "Fred", "Karen", "Moira", "Ava",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", nargs="+", default=ALL_VOICES)
    parser.add_argument("--speaker", type=str, default="tts_direct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/m10/raw") / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/m10/manifests") / "captures.jsonl"

    total = len(PROMPTS) * len(args.voices)
    print(f"M10 TTS Direct Capture — {total} captures ({len(PROMPTS)} prompts × {len(args.voices)} voices)")
    print(f"Output: {output_dir}")
    print()

    if args.dry_run:
        for v in args.voices:
            for p in PROMPTS:
                print(f"  [{v:10s}] {p['text']:15s} -> {p['label']}")
        return

    rows = []
    count = 0

    for voice in args.voices:
        print(f"\n=== {voice} ===")
        for p in PROMPTS:
            count += 1
            text, label = p["text"], p["label"]
            print(f"[{count}/{total}] '{text}'->{label}", end="", flush=True)

            try:
                audio = tts_to_audio(text, voice)
            except Exception as e:
                print(f" [ERR:{e}]")
                continue

            # Apply speaker→mic domain adaptation
            audio = sosfilt(_SPEAKER_LP, audio).astype(np.float32)

            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{label}_{voice[:3].lower()}.wav"
            wav_path = output_dir / filename
            write_pcm16_wav(wav_path, audio, sr=SR)

            row = {
                "capture_id": f"{voice[:3].lower()}_{timestamp[-6:]}",
                "session_id": session_id,
                "speaker": f"{args.speaker}_{voice}",
                "prompt": text, "label": label, "category": p.get("category", ""),
                "wav_path": str(wav_path.relative_to(Path("."))),
                "sample_rate": SR, "duration_s": round(len(audio) / SR, 3),
                "rms": rms, "peak": peak, "voice": voice,
                "source": "tts_direct", "created_at": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            q = "GOOD" if rms > 0.05 else "LOW" if rms > 0.02 else "SILENT"
            print(f"  rms={rms:.4f} [{q}]")

    # Summary
    by_label = {}
    for r in rows:
        l = r["label"]
        by_label.setdefault(l, []).append(r["rms"])

    print(f"\n{'='*60}")
    print(f"Session complete: {len(rows)} captures")
    for label in sorted(by_label):
        rms_list = by_label[label]
        print(f"  {label:4s}: {len(rms_list):3d} captures, avg rms={np.mean(rms_list):.4f}")


if __name__ == "__main__":
    main()
