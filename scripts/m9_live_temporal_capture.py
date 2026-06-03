#!/usr/bin/env python3
"""M9 live temporal capture.

Plays the saved temporal smoke sequence through the speaker path, captures the
mic response, and writes listenable WAV artifacts plus a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.live_capture import run_live_temporal_capture_report


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live temporal capture")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-temporal-capture"), help="Directory for rendered/captured WAV artifacts")
    parser.add_argument("--family", type=str, default="wide_sweep", choices=["default", "diphthong", "wide_sweep"], help="Temporal family to play into the mic")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the rendered temporal stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="flat", choices=["flat", "highpass"], help="Deterministic device profile to apply before capture")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice output device index")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    args = parser.parse_args()

    report = run_live_temporal_capture_report(
        args.output_dir,
        family=args.family,
        input_noise_rms=args.input_noise_rms,
        noise_seed=args.noise_seed,
        device_profile=args.device_profile,
        device_gain=args.device_gain,
        input_device=args.input_device,
        output_device=args.output_device,
        lead_in_s=args.lead_in_s,
        tail_s=args.tail_s,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
