#!/usr/bin/env python3
"""M10 consonant smoke harness.

Exercises the consonant decode path on a synthetic or captured audio buffer
and prints a structured JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.temporal import SR, FRAME_SIZE, write_pcm16_wav
from m9_sound.vowel_transitions import (
    VOWEL_FAMILIES,
    build_vowel_transition_report,
    decode_vowel_transition_audio,
    _synthesize_vowel_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 consonant smoke harness")
    parser.add_argument("--family", type=str, default="default", choices=sorted(VOWEL_FAMILIES.keys()))
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=sorted(VOWEL_FAMILIES.keys()))
    parser.add_argument("--noise-rms", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--device-profile", type=str, default="flat", choices=["flat", "highpass"])
    parser.add_argument("--device-gain", type=float, default=1.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--consonant-gate", type=float, default=2.0)
    parser.add_argument("--silence-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-wav", type=Path, default=None)
    args = parser.parse_args()

    report = build_vowel_transition_report(
        family=args.family,
        heldout_family=args.heldout_family,
        noise_rms=args.noise_rms,
        noise_seed=args.noise_seed,
        device_profile=args.device_profile,
        device_gain=args.device_gain,
        duration_scale=args.duration_scale,
    )
    report = dict(report)

    audio = np.asarray(report["rendered_audio"], dtype=np.float32)
    decode = decode_vowel_transition_audio(
        audio,
        consonant_gate=args.consonant_gate,
        silence_threshold=args.silence_threshold,
    )

    smoke_report = {
        "family": report["family"],
        "heldout_family": args.heldout_family,
        "consonant_gate": args.consonant_gate,
        "silence_threshold": args.silence_threshold,
        "consonant_count": int(decode["consonant_count"]),
        "consonant_rate": float(decode["consonant_rate"]),
        "vowel_segments": decode.get("vowel_segments", []),
        "decoded_transition_count": decode.get("decoded_transition_count", 0),
        "continuity": decode.get("continuity", 0.0),
        "frame_count": decode.get("frame_count", 0),
    }

    if args.output_wav is not None:
        meta = write_pcm16_wav(args.output_wav, audio, sr=SR)
        smoke_report["wav"] = str(meta["path"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")

    print(json.dumps(smoke_report, indent=2))


if __name__ == "__main__":
    main()
