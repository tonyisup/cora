#!/usr/bin/env python3
"""M10 live consonant demo.

Wait for the user to speak, detect consonant activity, and decode the
consonant/vowel/silence structure of what was said. Saves rendered/captured
WAVs and a JSON report.

Use --dry-run to exercise the full loop without hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.vowel_transitions import VOWEL_FAMILIES, build_vowel_transition_report, decode_vowel_transition_audio
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
) -> np.ndarray:
    if dry_run:
        listen_report = build_vowel_transition_report(
            family=listen_family,
            heldout_family=listen_family,
            noise_rms=0.0,
            noise_seed=7,
            device_profile="highpass",
            device_gain=1.0,
            sr=SR,
            frame_size=FRAME_SIZE,
        )
        listen_report = dict(listen_report)
        return np.asarray(cast(list[float], listen_report["rendered_audio"]), dtype=np.float32)

    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError("sounddevice is required for live capture") from exc

    sample_count = max(1, int(round(SR * listen_window_s)))
    recorded = sd.rec(sample_count, samplerate=SR, channels=1, device=input_device, dtype="float32")
    sd.wait()
    return np.asarray(recorded, dtype=np.float32).reshape(-1)


def _decode_and_report(
    audio: np.ndarray,
    *,
    consonant_gate: float,
    silence_threshold: float,
) -> dict[str, object]:
    """Decode audio with consonant awareness and return a structured report."""
    decode = decode_vowel_transition_audio(
        audio,
        consonant_gate=consonant_gate,
        silence_threshold=silence_threshold,
    )
    decode = dict(decode)

    labels: list[str] = list(cast(list[str], decode["decoded_vowels"]))
    consonant_frames = [i for i, l in enumerate(labels) if l == "C"]
    vowel_frames = [i for i, l in enumerate(labels) if l not in ("_", "C")]
    silence_frames = [i for i, l in enumerate(labels) if l == "_"]
    total = max(len(labels), 1)

    return {
        "labels": labels,
        "consonant_count": len(consonant_frames),
        "consonant_rate": len(consonant_frames) / total,
        "vowel_count": len(vowel_frames),
        "vowel_rate": len(vowel_frames) / total,
        "silence_count": len(silence_frames),
        "silence_rate": len(silence_frames) / total,
        "consonant_frames": consonant_frames[:20],
        "vowel_segments": decode.get("vowel_segments", []),
        "transition_count": decode.get("decoded_transition_count", 0),
        "continuity": decode.get("continuity", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M10 live consonant demo")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m10-live-consonant-demo"))
    parser.add_argument("--listen-family", type=str, default="phase_triad", choices=list(VOWEL_FAMILIES.keys()))
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--listen-window-s", type=float, default=0.6)
    parser.add_argument("--trigger-threshold", type=float, default=0.003)
    parser.add_argument("--min-speech-frames", type=int, default=2)
    parser.add_argument("--consonant-gate", type=float, default=2.0)
    parser.add_argument("--silence-threshold", type=float, default=0.001)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()

    if args.once and args.iterations == 0:
        args.iterations = 1

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M10 Live Consonant Demo")
    print("=" * 78)
    print("Speak into the microphone. CORA will detect consonants and vowels.")
    print(f"  Listen family:     {args.listen_family}")
    print(f"  Input device:      {args.input_device}")
    print(f"  Consonant gate:    {args.consonant_gate}")
    print(f"  Silence threshold: {args.silence_threshold}")
    print(f"  Listen window:     {args.listen_window_s:.2f}s")
    print(f"  Trigger threshold: {args.trigger_threshold}")
    print(f"  Output dir:        {output_root}")
    print(f"  Iterations:        {'forever' if args.iterations <= 0 else args.iterations}")
    print("  Ctrl+C to stop.\n")

    trigger_count = 0
    try:
        while args.iterations <= 0 or trigger_count < args.iterations:
            listen_audio = _listen_window_audio(
                input_device=args.input_device,
                listen_window_s=args.listen_window_s,
                dry_run=args.dry_run,
                listen_family=args.listen_family,
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
            print(f"[trigger {trigger_count:04d}] heard speech (rms={listen_rms:.4f}, frames={speech_frames})")

            report = _decode_and_report(
                listen_audio,
                consonant_gate=args.consonant_gate,
                silence_threshold=args.silence_threshold,
            )

            print(f"  consonant_rate={report['consonant_rate']:.1%} vowel_rate={report['vowel_rate']:.1%} silence_rate={report['silence_rate']:.1%}")
            print(f"  transitions={report['transition_count']} continuity={report['continuity']:.3f}")

            # Save artifacts
            iteration_dir = output_root / f"run-{trigger_count:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            wav_path = iteration_dir / "captured.wav"
            from m9_sound.temporal import write_pcm16_wav
            write_pcm16_wav(wav_path, listen_audio, sr=SR)
            report_path = iteration_dir / "report.json"
            report["trigger_idx"] = trigger_count
            report["wav"] = str(wav_path)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"  saved: {wav_path}")
            print(f"  report: {report_path}")

            if args.play:
                subprocess.run(["afplay", str(wav_path)], check=True)

            if args.pause_s > 0:
                time.sleep(args.pause_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"\nLoop summary: {trigger_count} triggers processed")
        latest = output_root / "report.json"
        if latest.exists():
            print(f"Latest report: {latest}")


if __name__ == "__main__":
    main()
