#!/usr/bin/env python3
"""M10 consonant corpus capture tool.

Records short consonant-bearing utterances and writes them to the M10 data
layout with a JSONL manifest entry per capture.

Usage:
    ./.venv/bin/python scripts/m10_record_consonants.py --speaker test_speaker
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.temporal import SR

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = REPO_ROOT / "data" / "m10_consonant_prompts.json"
DATA_ROOT = REPO_ROOT / "data" / "m10"


def _load_prompts() -> dict:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def _record_wav(
    prompt: str,
    speaker: str,
    session_id: str,
    input_device: int,
    duration_s: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Record audio from the microphone and return (audio, metadata)."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for live recording") from exc

    sample_count = int(round(SR * duration_s))
    print(f"  Recording '{prompt}' ({duration_s}s)...")
    recorded = sd.rec(sample_count, samplerate=SR, channels=1, device=input_device, dtype="float32")
    sd.wait()
    audio = np.asarray(recorded, dtype=np.float32).reshape(-1)

    input_info = sd.query_devices(input_device)
    meta = {
        "prompt": prompt,
        "speaker": speaker,
        "session_id": session_id,
        "input_device": input_device,
        "input_device_name": str(input_info.get("name", "")),
        "sample_rate": SR,
        "duration_s": round(len(audio) / SR, 3),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "peak": float(np.max(np.abs(audio))),
    }
    return audio, meta


def _save_capture(
    audio: np.ndarray,
    meta: dict,
    label: str,
    category: str,
    output_dir: Path,
) -> dict:
    """Save a WAV and return a manifest row."""
    from m9_sound.temporal import write_pcm16_wav

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    capture_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{meta['prompt']}_{capture_id}.wav"
    wav_path = output_dir / filename
    write_pcm16_wav(wav_path, audio, sr=SR)

    return {
        "capture_id": capture_id,
        "session_id": meta["session_id"],
        "speaker": meta["speaker"],
        "prompt": meta["prompt"],
        "label": label,
        "category": category,
        "wav_path": str(wav_path.relative_to(REPO_ROOT)),
        "sample_rate": meta["sample_rate"],
        "duration_s": meta["duration_s"],
        "rms": meta["rms"],
        "peak": meta["peak"],
        "input_device": meta["input_device"],
        "input_device_name": meta["input_device_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "m10_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 consonant corpus capture")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker identifier")
    parser.add_argument("--input-device", type=int, default=1, help="sounddevice input device index")
    parser.add_argument("--duration-s", type=float, default=1.0, help="Recording duration per prompt")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without recording")
    parser.add_argument("--auto", action="store_true", help="Auto-record without waiting for Enter (with countdown)")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH, help="Path to prompts JSON")
    args = parser.parse_args()

    prompts_data = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = prompts_data["prompts"]

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_ROOT / "raw" / args.speaker / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = DATA_ROOT / "manifests" / "captures.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M10 Consonant Corpus Capture")
    print("=" * 78)
    print(f"  Speaker:     {args.speaker}")
    print(f"  Session:     {session_id}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Input device: {args.input_device}")
    print(f"  Duration:    {args.duration_s}s per prompt")
    print(f"  Output dir:  {output_dir}")
    print(f"  Manifest:    {manifest_path}")
    print()

    if args.dry_run:
        print("Dry-run mode — prompts to record:")
        for p in prompts:
            print(f"  {p['token']:10s}  label={p['label']:4s}  category={p['category']}")
        return

    rows = []
    try:
        for idx, prompt_entry in enumerate(prompts):
            token = prompt_entry["token"]
            label = prompt_entry["label"]
            category = prompt_entry["category"]

            if args.auto:
                print(f"[{idx + 1}/{len(prompts)}] Say '{token}' ({label}, {category}) - recording in 1s...")
                time.sleep(1.0)
            else:
                print(f"[{idx + 1}/{len(prompts)}] Say '{token}' ({label}, {category})")
                print("  Press Enter to record (or 's' to skip, 'q' to quit)...")
                user_input = input("  > ").strip().lower()
                if user_input == "q":
                    break
                if user_input == "s":
                    print("  Skipped.")
                    continue

            audio, meta = _record_wav(token, args.speaker, session_id, args.input_device, args.duration_s)
            row = _save_capture(audio, meta, label, category, output_dir)
            rows.append(row)

            # Append to manifest
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(f"  Saved: {row['wav_path']}")
            print(f"  RMS: {row['rms']:.4f}  Peak: {row['peak']:.4f}")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"\nSession complete: {len(rows)} captures saved")
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
