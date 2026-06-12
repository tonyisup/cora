#!/usr/bin/env python3
"""M9 live voice imitation demo.

Wait for the user to speak, listen to the mic window, and then imitate with the
current M9 transition path. This is the simplest "speak, pause, CORA replies"
demo for showing the latest imitation progress.

Use --dry-run to exercise the same wait/listen/imitate loop without hardware.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_live_vowel_demo import _detect_speech, _listen_window_audio, _parse_family_list, _run_one_transition_trigger
from m9_sound import VOWEL_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live voice imitation demo")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-voice-imitate"), help="Directory for per-trigger rendered/captured WAV artifacts")
    parser.add_argument("--families", type=str, default="phase_triad,phase_arc", help="Comma-separated transition families CORA will imitate in rotation")
    parser.add_argument("--listen-family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()), help="Family used to synthesize dry-run mic speech")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the listened-to trigger audio")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="highpass", choices=["flat", "highpass"], help="Deterministic device profile to apply before capture")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=1, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=3, help="sounddevice output device index (Mac mini Speakers)")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--listen-window-s", type=float, default=0.6, help="How long to listen before deciding whether speech is present")
    parser.add_argument("--trigger-threshold", type=float, default=0.003, help="RMS threshold used to decide whether mic speech is present")
    parser.add_argument("--min-speech-frames", type=int, default=2, help="Minimum speech frames required before triggering imitation")
    parser.add_argument("--pause-s", type=float, default=0.5, help="Pause between loop iterations and after each imitation")
    parser.add_argument("--iterations", type=int, default=0, help="Number of imitation triggers to run; 0 means run forever")
    parser.add_argument("--once", action="store_true", help="Convenience flag: run a single speak->listen->imitate cycle")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware and use a synthetic mic-trigger sequence")
    args = parser.parse_args()

    if args.once and args.iterations == 0:
        args.iterations = 1

    families = _parse_family_list(args.families)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M9 Live Voice Imitation Demo")
    print("=" * 78)
    print("Speak into the microphone. CORA will wait, listen, and then imitate.")
    print(f"  Families:          {', '.join(families)}")
    print(f"  Listen family:     {args.listen_family}")
    print(f"  Input device:      {args.input_device}")
    print(f"  Output device:     {args.output_device}")
    print(f"  Device profile:    {args.device_profile}")
    print(f"  Listen window:     {args.listen_window_s:.2f}s")
    print(f"  Trigger threshold: {args.trigger_threshold}")
    print(f"  Output dir:        {output_root}")
    print(f"  Iterations:        {'forever' if args.iterations <= 0 else args.iterations}")
    print("  Ctrl+C to stop.\n")

    trigger_count = 0
    passed = 0
    family_index = 0
    try:
        while args.iterations <= 0 or trigger_count < args.iterations:
            listen_audio = _listen_window_audio(
                input_device=args.input_device,
                listen_window_s=args.listen_window_s,
                dry_run=args.dry_run,
                listen_family=args.listen_family,
                input_noise_rms=args.input_noise_rms,
                noise_seed=args.noise_seed,
                device_profile=args.device_profile,
                device_gain=args.device_gain,
            )
            heard, listen_rms, speech_frames = _detect_speech(
                listen_audio,
                threshold=args.trigger_threshold,
                min_speech_frames=args.min_speech_frames,
            )
            if not heard:
                print(f"[listen] no trigger (rms={listen_rms:.4f}, speech_frames={speech_frames})")
                if args.pause_s > 0:
                    time.sleep(args.pause_s)
                continue

            trigger_count += 1
            family = families[family_index % len(families)]
            family_index += 1
            print(f"[trigger {trigger_count:04d}] heard speech (rms={listen_rms:.4f}, frames={speech_frames}) -> imitate {family}")
            report = _run_one_transition_trigger(
                trigger_idx=trigger_count,
                family=family,
                output_root=output_root,
                heldout_family=args.listen_family,
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
            phase = report["phase_closure"]
            score = float(phase["score"])
            passed_flag = bool(phase["passed"])
            if passed_flag:
                passed += 1
            print(f"          score={score:.3f} passed={passed_flag} captured={report['captured_decoded_transition_count']}")
            print(f"          rendered={report['rendered_wav']}")
            print(f"          captured={report['captured_wav']}")
            if args.pause_s > 0:
                time.sleep(args.pause_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"\nLoop summary: {passed}/{trigger_count} triggers passed")
        latest_report = output_root / "report.json"
        if latest_report.exists():
            print(f"Latest report: {latest_report}")


if __name__ == "__main__":
    main()
