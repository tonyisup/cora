#!/usr/bin/env python3
"""M9 vowel-transition family smoke harness.

Prints a structured JSON report for the requested family and can optionally
write a listenable WAV artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.vowel_transitions import build_vowel_transition_report, write_vowel_transition_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 vowel-transition family smoke harness")
    parser.add_argument("--family", type=str, default="default", choices=["default", "diphthong", "wide_sweep"])
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=["default", "diphthong", "wide_sweep"])
    parser.add_argument("--noise-rms", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--device-profile", type=str, default="flat", choices=["flat", "highpass"])
    parser.add_argument("--device-gain", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the JSON report")
    parser.add_argument("--output-wav", type=Path, default=None, help="Optional path to write the rendered WAV")
    args = parser.parse_args()

    if args.output_wav is not None:
        report = write_vowel_transition_wav(
            args.output_wav,
            family=args.family,
            heldout_family=args.heldout_family,
            noise_rms=args.noise_rms,
            noise_seed=args.noise_seed,
            device_profile=args.device_profile,
            device_gain=args.device_gain,
        )
    else:
        report = build_vowel_transition_report(
            family=args.family,
            heldout_family=args.heldout_family,
            noise_rms=args.noise_rms,
            noise_seed=args.noise_seed,
            device_profile=args.device_profile,
            device_gain=args.device_gain,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
