#!/usr/bin/env python3
"""M9 live vowel-transition capture.

Plays a vowel-transition family through the speaker path, captures the mic
response, and writes listenable WAV artifacts plus a JSON report.

Use --dry-run to exercise the full artifact path without hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound import VOWEL_FAMILIES, run_live_vowel_transition_capture_report
from m9_sound.live_capture import LIVE_VOWEL_TRANSITION_PRESETS


def _dry_run_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    tail = np.zeros(sr // 20, dtype=np.float32)
    return np.concatenate([np.asarray(playback_audio, dtype=np.float32).reshape(-1), tail])


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel-transition capture")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-vowel-transition-capture"), help="Directory for rendered/captured WAV artifacts")
    parser.add_argument("--family", type=str, default="default", choices=list(VOWEL_FAMILIES.keys()), help="Vowel-transition family to play into the mic")
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=list(VOWEL_FAMILIES.keys()), help="Held-out family used in the score report")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the rendered vowel stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="flat", choices=["flat", "highpass"], help="Deterministic device profile to apply before capture")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=3, help="sounddevice output device index (Mac mini Speakers)")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Scale family durations for live tuning")
    parser.add_argument("--preset", type=str, default="default", choices=sorted(LIVE_VOWEL_TRANSITION_PRESETS.keys()), help="Named live-tuning preset")
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
        duration_scale=args.duration_scale,
        preset=args.preset,
        capture_audio_fn=_dry_run_capture_audio if args.dry_run else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
