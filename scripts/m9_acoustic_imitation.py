#!/usr/bin/env python3
"""M9 Acoustic Imitation Loop — with REINFORCE learning.

Simulates the real-world loop:
1. Play a "human" stimulus sound from device A (speaker)
2. Capture it through mic → stimulus_capture (the "target")
3. Producer generates an imitation
4. Play imitation from device B (headphones)
5. Capture it through mic → imitation_capture
6. Score imitation_capture vs stimulus_capture → reward
7. REINFORCE update: adjust producer weights to increase reward
8. Repeat for multiple trials

The producer learns to imitate the stimulus by maximizing the similarity
between its own captured output and the stimulus capture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticProducer, AcousticSimilarity, OscillatorBank

SR = 16000


def play_and_capture(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    sr: int = SR,
    lead_in_s: float = 0.25,
    tail_s: float = 0.25,
) -> dict[str, Any]:
    """Play audio on output_device and capture from input_device."""
    import sounddevice as sd

    if pcm.ndim != 1:
        pcm = np.asarray(pcm).reshape(-1)
    pcm = pcm.astype(np.float32)

    input_info = sd.query_devices(input_device)
    input_sr = int(input_info["default_samplerate"] or sr)
    input_name = str(input_info["name"])
    output_name = str(sd.query_devices(output_device)["name"])

    playback_duration = len(pcm) / sr
    total_duration = lead_in_s + playback_duration + tail_s
    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)

    lead_in_samples = int(lead_in_s * input_sr)
    pos = 0
    playback_started_at: float | None = None
    capture_started_at = time.perf_counter()

    stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=input_sr,
        dtype="float32",
        blocksize=1024,
    )
    stream.start()

    try:
        while pos < total_samples:
            if playback_started_at is None and pos >= lead_in_samples:
                sd.play(pcm, samplerate=sr, device=output_device, blocking=False)
                playback_started_at = time.perf_counter()
            data, _ = stream.read(min(1024, total_samples - pos))
            n = min(len(data), total_samples - pos)
            captured[pos : pos + n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()

    if input_sr != sr:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, sr, input_sr).astype(np.float32)

    return {
        "captured_pcm": captured.astype(np.float32),
        "input_device_name": input_name,
        "output_device_name": output_name,
        "capture_duration_s": round(total_duration, 4),
        "playback_duration_s": round(playback_duration, 4),
        "capture_elapsed_s": round(time.perf_counter() - capture_started_at, 4),
        "playback_started": playback_started_at is not None,
    }


def generate_stimulus(
    freq_hz: float,
    duration_s: float,
    waveform: str = "sine",
    sr: int = SR,
) -> np.ndarray:
    """Generate a pure-tone stimulus sound (the "human" sound)."""
    osc = OscillatorBank(sr=sr)
    fn = {
        "sine": osc.sine,
        "triangle": osc.triangle,
        "sawtooth": osc.sawtooth,
        "square": osc.square,
    }[waveform]
    return fn(freq_hz, duration_s, amplitude=0.5).astype(np.float32)


def reinforce_step(
    controller,
    hidden: np.ndarray,
    reward: float,
    baseline: float,
    lr: float = 0.1,
) -> None:
    """REINFORCE update: increase weights for actions above baseline."""
    advantage = reward - baseline
    if advantage <= 0:
        return

    h = hidden.flatten()[:controller._hidden_dim]
    if len(h) < controller._hidden_dim:
        h = np.pad(h, (0, controller._hidden_dim - len(h)))

    controller._W += lr * advantage * np.outer(np.ones(4), h)
    controller._b += lr * advantage * np.ones(4) * 0.5
    controller.clip_weights(max_norm=5.0)


def run_imitation_loop(
    *,
    stimulus_freq_hz: float,
    stimulus_waveform: str,
    stimulus_device: int,
    imitation_device: int,
    input_device: int,
    duration_s: float,
    n_trials: int,
    lr: float,
    sr: int = SR,
) -> dict[str, Any]:
    """Run the full acoustic imitation loop with REINFORCE learning."""
    similarity = AcousticSimilarity(sr=sr)

    # 1. Generate stimulus (the "human" sound)
    stimulus_pcm = generate_stimulus(stimulus_freq_hz, duration_s, stimulus_waveform, sr)

    # 2. Play stimulus on device A, capture through mic
    print(f"  Playing stimulus ({stimulus_freq_hz:.1f} Hz {stimulus_waveform}) on device {stimulus_device}...")
    stimulus_result = play_and_capture(
        stimulus_pcm,
        input_device=input_device,
        output_device=stimulus_device,
        sr=sr,
    )
    stimulus_capture = stimulus_result["captured_pcm"]

    # Self-score baseline
    self_score = similarity.detailed_score(stimulus_pcm, stimulus_capture)
    print(f"  Stimulus self-score: {self_score.overall:.4f}")

    # Initialize producer and baseline
    producer = AcousticProducer(sr=sr, hidden_dim=32)
    baseline = 0.0
    trial_rows = []

    for trial_idx in range(n_trials):
        # 3. Producer generates imitation
        imitation_pcm = producer.produce(duration_s)

        # 4. Play imitation on device B, capture through mic
        imitation_result = play_and_capture(
            imitation_pcm,
            input_device=input_device,
            output_device=imitation_device,
            sr=sr,
        )
        imitation_capture = imitation_result["captured_pcm"]

        # 5. Score: how well does the imitation match the stimulus?
        score = similarity.detailed_score(stimulus_capture, imitation_capture)
        reward = float(score.overall)

        # 6. Update baseline
        baseline = 0.9 * baseline + 0.1 * reward

        # 7. REINFORCE update
        reinforce_step(
            producer.controller,
            producer.substrate.hidden_state,
            reward,
            baseline,
            lr=lr,
        )

        trial_rows.append({
            "trial": trial_idx + 1,
            "reward": reward,
            "baseline": float(baseline),
            "mel": float(score.mfcc_distance),
            "pitch": float(score.pitch_distance),
            "duration": float(score.duration_distance),
            "loudness": float(score.loudness_distance),
        })

        print(f"  Trial {trial_idx + 1:3d}/{n_trials}  reward={reward:.4f}  "
              f"baseline={baseline:.4f}  mel={score.mfcc_distance:.4f}  "
              f"pitch={score.pitch_distance:.4f}")

    return {
        "stimulus": {
            "freq_hz": stimulus_freq_hz,
            "waveform": stimulus_waveform,
            "duration_s": duration_s,
            "device_name": stimulus_result["output_device_name"],
        },
        "imitation": {
            "device_name": imitation_result["output_device_name"],
        },
        "config": {
            "n_trials": n_trials,
            "lr": lr,
            "hidden_dim": 32,
        },
        "self_score": {
            "overall": float(self_score.overall),
            "mel": float(self_score.mfcc_distance),
            "pitch": float(self_score.pitch_distance),
            "duration": float(self_score.duration_distance),
            "loudness": float(self_score.loudness_distance),
        },
        "trials": trial_rows,
        "summary": {
            "first_reward": trial_rows[0]["reward"] if trial_rows else 0,
            "last_reward": trial_rows[-1]["reward"] if trial_rows else 0,
            "best_reward": max(t["reward"] for t in trial_rows) if trial_rows else 0,
            "improvement": (trial_rows[-1]["reward"] - trial_rows[0]["reward"]) if trial_rows else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 acoustic imitation loop with learning")
    parser.add_argument("--stimulus-freq", type=float, default=220.0)
    parser.add_argument("--stimulus-waveform", type=str, default="sine",
                        choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--stimulus-device", type=int, default=3)
    parser.add_argument("--imitation-device", type=int, default=2)
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="/tmp/m9_acoustic_imitation_learning.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Acoustic Imitation Loop — REINFORCE Learning")
    print("=" * 72)
    print(f"  Stimulus: {args.stimulus_freq:.1f} Hz {args.stimulus_waveform}")
    print(f"  Duration: {args.duration:.2f}s")
    print(f"  Trials: {args.trials}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Stimulus device: {args.stimulus_device} (speaker)")
    print(f"  Imitation device: {args.imitation_device} (headphones)")
    print(f"  Input device: {args.input_device} (mic)")
    print()

    result = run_imitation_loop(
        stimulus_freq_hz=args.stimulus_freq,
        stimulus_waveform=args.stimulus_waveform,
        stimulus_device=args.stimulus_device,
        imitation_device=args.imitation_device,
        input_device=args.input_device,
        duration_s=args.duration,
        n_trials=args.trials,
        lr=args.lr,
    )

    summary = result["summary"]
    print()
    print("Summary")
    print("-" * 72)
    print(f"  First reward:  {summary['first_reward']:.4f}")
    print(f"  Last reward:   {summary['last_reward']:.4f}")
    print(f"  Best reward:   {summary['best_reward']:.4f}")
    print(f"  Improvement:   {summary['improvement']:+.4f}")
    print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved to: {args.output}")

    if summary["improvement"] > 0:
        print("  STATUS: producer improved ✓")
    else:
        print("  STATUS: no improvement (may need more trials or higher lr)")


if __name__ == "__main__":
    main()
