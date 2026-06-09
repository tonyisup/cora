#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""M9 live vowel imitation demo using the latest validated live transition path.

This is the current main demo for live mic imitation.
It renders/captures the latest phase-closure family path and writes a JSON
report plus listenable WAV artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound import VOWEL_FAMILIES, run_live_vowel_transition_capture_report


def _dry_run_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    tail = np.zeros(sr // 20, dtype=np.float32)
    return np.concatenate([np.asarray(playback_audio, dtype=np.float32).reshape(-1), tail])


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel imitation demo (latest live transition path)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-vowel-demo"), help="Directory for rendered/captured WAV artifacts")
    parser.add_argument("--family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()), help="Vowel-transition family to play into the mic")
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=list(VOWEL_FAMILIES.keys()), help="Held-out family used in the score report")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the rendered vowel stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="highpass", choices=["flat", "highpass"], help="Deterministic device profile to apply before capture")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=1, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=2, help="sounddevice output device index")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware and use a synthetic capture tail")
    args = parser.parse_args()

    report = run_live_vowel_transition_capture_report(
        args.output_dir,
        family=args.family,
        heldout_family=args.heldout_family,
        input_noise_rms=args.input_noise_rms,
        noise_seed=args.noise_seed,
        device_profile=args.device_profile,
        device_gain=args.device_gain,
        input_device=args.input_device,
        output_device=args.output_device,
        lead_in_s=args.lead_in_s,
        tail_s=args.tail_s,
        capture_audio_fn=_dry_run_capture_audio if args.dry_run else None,
    )

    summary = report["phase_closure"]
    print(
        f"Live vowel demo: family={report['family']} score={summary['score']:.3f} "
        f"passed={summary['passed']} target={summary['target_transition_count']} "
        f"captured={report['captured_decoded_transition_count']}"
    )
    print(f"Rendered WAV: {report['rendered_wav']}")
    print(f"Captured WAV: {report['captured_wav']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
