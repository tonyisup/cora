#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""M9 live vowel imitation loop using the latest validated live transition path.

This is the current main demo for live mic imitation.
It stays running, repeatedly exercising the latest phase-closure path and
writing per-iteration WAV artifacts plus a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound import VOWEL_FAMILIES, run_live_vowel_transition_capture_report


def _dry_run_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    tail = np.zeros(sr // 20, dtype=np.float32)
    return np.concatenate([np.asarray(playback_audio, dtype=np.float32).reshape(-1), tail])


def _run_one_iteration(
    *,
    iteration: int,
    output_root: Path,
    family: str,
    heldout_family: str,
    input_noise_rms: float,
    noise_seed: int | None,
    device_profile: str,
    device_gain: float,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
    dry_run: bool,
) -> dict[str, object]:
    iteration_dir = output_root / f"run-{iteration:04d}"
    report = run_live_vowel_transition_capture_report(
        iteration_dir,
        family=family,
        heldout_family=heldout_family,
        input_noise_rms=input_noise_rms,
        noise_seed=noise_seed,
        device_profile=device_profile,
        device_gain=device_gain,
        input_device=input_device,
        output_device=output_device,
        lead_in_s=lead_in_s,
        tail_s=tail_s,
        capture_audio_fn=_dry_run_capture_audio if dry_run else None,
    )
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel imitation loop (latest live transition path)")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-vowel-demo"), help="Directory for per-iteration rendered/captured WAV artifacts")
    parser.add_argument("--family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()), help="Vowel-transition family to play into the mic each iteration")
    parser.add_argument("--heldout-family", type=str, default="diphthong", choices=list(VOWEL_FAMILIES.keys()), help="Held-out family used in the score report")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the rendered vowel stimulus")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="highpass", choices=["flat", "highpass"], help="Deterministic device profile to apply before capture")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=1, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=2, help="sounddevice output device index")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--pause-s", type=float, default=0.5, help="Pause between loop iterations")
    parser.add_argument("--iterations", type=int, default=0, help="Number of iterations to run; 0 means run forever")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware and use a synthetic capture tail")
    args = parser.parse_args()

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("M9 Live Vowel Imitation Loop")
    print("=" * 72)
    print(f"  Family:          {args.family}")
    print(f"  Heldout family:  {args.heldout_family}")
    print(f"  Device profile:  {args.device_profile}")
    print(f"  Input device:    {args.input_device}")
    print(f"  Output device:   {args.output_device}")
    print(f"  Output dir:      {output_root}")
    print(f"  Iterations:      {'forever' if args.iterations <= 0 else args.iterations}")
    print(f"  Pause:           {args.pause_s:.2f}s")
    print("  Ctrl+C to stop.\n")

    completed = 0
    passed = 0
    try:
        while args.iterations <= 0 or completed < args.iterations:
            completed += 1
            report = _run_one_iteration(
                iteration=completed,
                output_root=output_root,
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
                dry_run=args.dry_run,
            )
            summary = report["phase_closure"]
            summary_dict = summary if isinstance(summary, dict) else {}
            score = float(summary_dict.get("score", 0.0))
            passed_flag = bool(summary_dict.get("passed", False))
            if passed_flag:
                passed += 1
            print(
                f"[{completed:04d}] family={report['family']} score={score:.3f} "
                f"passed={passed_flag} captured={report['captured_decoded_transition_count']} "
                f"-> {report['output_dir']}"
            )
            print(f"      rendered={report['rendered_wav']}")
            print(f"      captured={report['captured_wav']}")
            if args.pause_s > 0 and (args.iterations <= 0 or completed < args.iterations):
                time.sleep(args.pause_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"\nLoop summary: {passed}/{completed} iterations passed")
        if completed:
            latest_report = output_root / "report.json"
            if latest_report.exists():
                print(f"Latest report: {latest_report}")


if __name__ == "__main__":
    main()
