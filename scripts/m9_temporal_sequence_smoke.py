#!/usr/bin/env python3
"""M9 temporal-sequence smoke harness.

Runs the saved temporal controller/substrate pair on a short tone sequence and
prints a compact JSON report. This is hardware-free and suitable for regression
checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.temporal import run_temporal_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 temporal-sequence smoke harness")
    parser.add_argument("--family", type=str, default="default", choices=["default", "diphthong"], help="Temporal tone family to smoke test")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    args = parser.parse_args()

    report = run_temporal_smoke_report(
        family=args.family,
        input_noise_rms=args.input_noise_rms,
        noise_seed=args.noise_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
