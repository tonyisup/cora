#!/usr/bin/env python3
"""M9 live vowel-transition phase-closure gate.

Runs the live vowel-transition capture wrapper across one or more families and
aggregates pass rate / score summaries for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.live_capture import LIVE_VOWEL_TRANSITION_PRESETS, run_live_vowel_transition_gate
from m9_sound.vowel_transitions import VOWEL_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel-transition phase-closure gate")
    parser.add_argument("--output", type=Path, default=Path("/tmp/m9_live_vowel_transition_gate.json"), help="Where to write the JSON report")
    parser.add_argument("--families", type=str, default="default,phase_triad,phase_arc", help="Comma-separated vowel-transition families to gate")
    parser.add_argument("--repeats", type=int, default=2, help="How many repeats per family")
    parser.add_argument("--threshold", type=float, default=0.7, help="Phase-closure score threshold")
    parser.add_argument("--dry-run", action="store_true", help="Use a synthetic capture path instead of hardware")
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=list(VOWEL_FAMILIES.keys()), help="Held-out family used in per-capture scoring")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Deterministic input noise injected into the rendered stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when injecting input noise")
    parser.add_argument("--device-profile", type=str, default="flat", choices=["flat", "highpass"], help="Deterministic device profile to apply")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=3, help="sounddevice output device index (Mac mini Speakers)")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Scale the family duration for live tuning")
    parser.add_argument("--preset", type=str, default="default", choices=sorted(LIVE_VOWEL_TRANSITION_PRESETS.keys()), help="Named live-tuning preset")
    args = parser.parse_args()

    families = [family.strip() for family in args.families.split(",") if family.strip()]
    report = run_live_vowel_transition_gate(
        args.output.parent,
        families=families,
        repeats=args.repeats,
        threshold=args.threshold,
        dry_run=args.dry_run,
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
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
