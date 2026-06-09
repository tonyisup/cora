#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""M9 live vowel-transition imitation loop.

This is the main demo for live mic imitation.
It waits for mic speech, then runs the latest vowel-transition capture path so
what gets imitated is a transition family, not just a single vowel.

Use --dry-run to exercise the same waiting + imitation loop without hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound import VOWEL_FAMILIES, build_vowel_transition_report, run_live_vowel_transition_capture_report
from m9_sound.temporal import FRAME_SIZE, SR

HP_SOS = butter(4, 100, btype="high", fs=SR, output="sos")


def _dry_run_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
    tail = np.zeros(sr // 20, dtype=np.float32)
    return np.concatenate([np.asarray(playback_audio, dtype=np.float32).reshape(-1), tail])


def _detect_speech(audio: np.ndarray, *, threshold: float, min_speech_frames: int) -> tuple[bool, float, int]:
    filtered = np.asarray(sosfilt(HP_SOS, np.asarray(audio, dtype=np.float32).reshape(-1)), dtype=np.float32)
    frame_count = max(1, len(filtered) // FRAME_SIZE)
    speech_frames = 0
    for idx in range(frame_count):
        frame = filtered[idx * FRAME_SIZE : (idx + 1) * FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            continue
        frame_rms = float(np.sqrt(np.mean(frame**2)))
        if frame_rms >= threshold:
            speech_frames += 1
    rms = float(np.sqrt(np.mean(filtered**2))) if len(filtered) else 0.0
    return speech_frames >= min_speech_frames, rms, speech_frames


def _listen_window_audio(
    *,
    input_device: int,
    listen_window_s: float,
    dry_run: bool,
    listen_family: str,
    input_noise_rms: float,
    noise_seed: int | None,
    device_profile: str,
    device_gain: float,
) -> np.ndarray:
    if dry_run:
        listen_report = build_vowel_transition_report(
            family=listen_family,
            heldout_family=listen_family,
            noise_rms=input_noise_rms,
            noise_seed=noise_seed,
            device_profile=device_profile,
            device_gain=device_gain,
            sr=SR,
            frame_size=FRAME_SIZE,
        )
        return np.asarray(cast(list[float], listen_report["rendered_audio"]), dtype=np.float32)

    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - only on live hardware
        raise RuntimeError("sounddevice is required for live capture; install the optional audio deps") from exc

    sample_count = max(1, int(round(SR * listen_window_s)))
    recorded = sd.rec(sample_count, samplerate=SR, channels=1, device=input_device, dtype="float32")
    sd.wait()
    return np.asarray(recorded, dtype=np.float32).reshape(-1)


def _run_one_transition_trigger(
    *,
    trigger_idx: int,
    family: str,
    output_root: Path,
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
    iteration_dir = output_root / f"run-{trigger_idx:04d}"
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
    report = dict(report)
    report["trigger_idx"] = trigger_idx
    report["family"] = family
    report["iteration_dir"] = str(iteration_dir)
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_family_list(text: str) -> list[str]:
    families = [item.strip() for item in text.split(",") if item.strip()]
    if not families:
        raise ValueError("At least one vowel-transition family is required")
    for family in families:
        if family not in VOWEL_FAMILIES:
            raise ValueError(f"Unknown vowel-transition family: {family}")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 live vowel-transition imitation loop")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-live-vowel-demo"), help="Directory for per-trigger rendered/captured WAV artifacts")
    parser.add_argument("--families", type=str, default="phase_triad,phase_arc", help="Comma-separated vowel-transition families to imitate in rotation")
    parser.add_argument("--listen-family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()), help="Family used to synthesize dry-run mic speech")
    parser.add_argument("--input-noise-rms", type=float, default=0.0, help="Add deterministic Gaussian noise to the listened-to trigger audio")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed used when adding input noise")
    parser.add_argument("--device-profile", type=str, default="highpass", choices=["flat", "highpass"], help="Device profile used for the transition render")
    parser.add_argument("--device-gain", type=float, default=1.0, help="Gain applied as part of the device profile")
    parser.add_argument("--input-device", type=int, default=1, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=2, help="sounddevice output device index")
    parser.add_argument("--lead-in-s", type=float, default=0.25, help="Silent lead-in before playback")
    parser.add_argument("--tail-s", type=float, default=0.25, help="Silent tail after playback")
    parser.add_argument("--listen-window-s", type=float, default=0.6, help="How long to listen before deciding whether speech is present")
    parser.add_argument("--trigger-threshold", type=float, default=0.003, help="RMS threshold used to decide whether mic speech is present")
    parser.add_argument("--min-speech-frames", type=int, default=2, help="Minimum speech frames required before triggering imitation")
    parser.add_argument("--pause-s", type=float, default=0.5, help="Pause between loop iterations and after each imitation")
    parser.add_argument("--iterations", type=int, default=0, help="Number of imitation triggers to run; 0 means run forever")
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware and use a synthetic mic-trigger sequence")
    args = parser.parse_args()

    families = _parse_family_list(args.families)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M9 Live Vowel-Transition Imitation Loop")
    print("=" * 78)
    print(f"  Families:         {', '.join(families)}")
    print(f"  Listen family:    {args.listen_family}")
    print(f"  Input device:     {args.input_device}")
    print(f"  Output device:    {args.output_device}")
    print(f"  Device profile:   {args.device_profile}")
    print(f"  Listen window:    {args.listen_window_s:.2f}s")
    print(f"  Trigger threshold: {args.trigger_threshold}")
    print(f"  Output dir:       {output_root}")
    print(f"  Iterations:       {'forever' if args.iterations <= 0 else args.iterations}")
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
            print(
                f"[trigger {trigger_count:04d}] heard speech (rms={listen_rms:.4f}, frames={speech_frames}) -> imitate {family}"
            )
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
            phase = cast(dict[str, object], report["phase_closure"])
            score = float(cast(float, phase["score"]))
            passed_flag = bool(cast(bool, phase["passed"]))
            if passed_flag:
                passed += 1
            print(
                f"          score={score:.3f} passed={passed_flag} captured={report['captured_decoded_transition_count']}"
            )
            print(f"          rendered={report['rendered_wav']}")
            print(f"          captured={report['captured_wav']}")
            if args.pause_s > 0:
                time.sleep(args.pause_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"\nLoop summary: {passed}/{trigger_count} transition triggers passed")
        latest_report = output_root / "report.json"
        if latest_report.exists():
            print(f"Latest report: {latest_report}")


if __name__ == "__main__":
    main()
