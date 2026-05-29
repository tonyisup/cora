#!/usr/bin/env python3
"""M9 Acoustic Imitation Loop — self-listening with REINFORCE learning.

Two modes:
1. Single frequency: --stimulus-freq 220
2. Curriculum: --curriculum 220,262,330,392,440

The loop:
1. Play stimulus on device A (speaker), capture through mic
2. Producer generates imitation, play on device B (headphones), capture through mic
3. Score imitation_capture vs stimulus_capture → reward
4. REINFORCE update
5. Save checkpoint for next run / next frequency
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
    substrate_hidden: np.ndarray,
    reward: float,
    baseline: float,
    lr: float = 0.1,
) -> None:
    """REINFORCE update through two-layer controller.

    Computes hidden activation from substrate hidden state, then updates
    W2, b2 (output layer) and W1, b1 (hidden layer) to increase reward.
    """
    advantage = reward - baseline
    if advantage <= 0:
        return

    h = substrate_hidden.flatten()[:controller._hidden_dim]
    if len(h) < controller._hidden_dim:
        h = np.pad(h, (0, controller._hidden_dim - len(h)))

    # Forward pass to get hidden activation
    raw, a1 = controller._forward_all(h)
    n_out = 4 + controller._n_waveforms

    # Gradient for output layer (W2, b2): d(output)/d(W2) = a1
    # Update: increase output for above-baseline rewards
    delta = lr * advantage
    controller._W2 += delta * np.outer(np.ones(n_out), a1) * 0.1
    controller._b2[:4] += delta * np.ones(4) * 0.05  # only update param logits

    # Backprop to hidden layer: d(output)/d(a1) = W2, d(a1)/d(z1) = 1 - tanh^2(z1)
    z1 = controller._W1 @ h + controller._b1
    da1 = controller._W2.T @ (np.ones(n_out) * delta * 0.1)
    dz1 = da1 * (1.0 - np.tanh(z1) ** 2)  # tanh derivative

    controller._W1 += np.outer(dz1, h) * 0.1
    controller._b1 += dz1 * 0.05

    controller.clip_weights(max_norm=5.0)


def parse_freqs(spec: str) -> list[float]:
    """Parse comma-separated frequency list like '220,262,330,392,440'."""
    return [float(f.strip()) for f in spec.split(",") if f.strip()]


def run_single_freq(
    *,
    stimulus_freq_hz: float,
    stimulus_waveform: str,
    stimulus_device: int,
    imitation_device: int,
    input_device: int,
    duration_s: float,
    n_trials: int,
    lr: float,
    checkpoint_path: str | None,
    substrate_checkpoint_path: str | None = None,
    sr: int = SR,
) -> dict[str, Any]:
    """Run imitation loop on a single frequency."""
    similarity = AcousticSimilarity(sr=sr)
    producer = AcousticProducer(sr=sr, hidden_dim=32)
    baseline = 0.0

    # Load producer checkpoint if provided and exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading producer checkpoint: {checkpoint_path}")
        producer.load_checkpoint(checkpoint_path)

    # Load pre-trained substrate weights if provided
    if substrate_checkpoint_path and Path(substrate_checkpoint_path).exists():
        print(f"  Loading pre-trained substrate: {substrate_checkpoint_path}")
        import json as _json
        sub_data = _json.loads(Path(substrate_checkpoint_path).read_text())
        producer.substrate._W_h = np.array(sub_data["W_h"], dtype=np.float32)
        producer.substrate._W_x = np.array(sub_data["W_x"], dtype=np.float32)
        producer.substrate._b = np.array(sub_data["b"], dtype=np.float32)

    # Generate and play stimulus
    stimulus_pcm = generate_stimulus(stimulus_freq_hz, duration_s, stimulus_waveform, sr)
    print(f"  Playing stimulus ({stimulus_freq_hz:.1f} Hz {stimulus_waveform})...")
    stimulus_result = play_and_capture(
        stimulus_pcm, input_device=input_device, output_device=stimulus_device, sr=sr,
    )
    stimulus_capture = stimulus_result["captured_pcm"]

    self_score = similarity.detailed_score(stimulus_pcm, stimulus_capture)
    print(f"  Stimulus self-score: {self_score.overall:.4f}")

    # Pre-encode stimulus: feed captured audio through the substrate so
    # the hidden state encodes what was heard. The controller then uses
    # this representation to know what to imitate.
    n_frames = len(stimulus_capture) // producer._frame_size
    for i in range(n_frames):
        frame = stimulus_capture[i * producer._frame_size:(i + 1) * producer._frame_size]
        producer.substrate.step(frame)

    trial_rows = []
    for trial_idx in range(n_trials):
        imitation_pcm = producer.produce(duration_s)

        imitation_result = play_and_capture(
            imitation_pcm, input_device=input_device, output_device=imitation_device, sr=sr,
        )
        imitation_capture = imitation_result["captured_pcm"]

        score = similarity.detailed_score(stimulus_capture, imitation_capture)
        reward = float(score.overall)
        baseline = 0.9 * baseline + 0.1 * reward

        reinforce_step(producer.controller, producer.substrate.hidden_state, reward, baseline, lr=lr)

        trial_rows.append({
            "trial": trial_idx + 1,
            "reward": reward,
            "baseline": float(baseline),
            "mel": float(score.mfcc_distance),
            "pitch": float(score.pitch_distance),
            "duration": float(score.duration_distance),
            "loudness": float(score.loudness_distance),
        })

        if (trial_idx + 1) % 10 == 0:
            print(f"  Trial {trial_idx + 1:3d}/{n_trials}  "
                  f"reward={reward:.4f}  baseline={baseline:.4f}")

    if checkpoint_path:
        producer.save_checkpoint(checkpoint_path)

    return {
        "mode": "single",
        "stimulus": {"freq_hz": stimulus_freq_hz, "waveform": stimulus_waveform},
        "self_score": {"overall": float(self_score.overall)},
        "trials": trial_rows,
        "summary": {
            "first_reward": trial_rows[0]["reward"] if trial_rows else 0,
            "last_reward": trial_rows[-1]["reward"] if trial_rows else 0,
            "best_reward": max(t["reward"] for t in trial_rows) if trial_rows else 0,
            "improvement": (trial_rows[-1]["reward"] - trial_rows[0]["reward"]) if trial_rows else 0,
        },
    }


def run_curriculum(
    *,
    freqs: list[float],
    stimulus_waveform: str,
    stimulus_device: int,
    imitation_device: int,
    input_device: int,
    duration_s: float,
    trials_per_freq: int,
    lr: float,
    checkpoint_path: str | None,
    substrate_checkpoint_path: str | None = None,
    sr: int = SR,
) -> dict[str, Any]:
    """Train on each frequency sequentially, accumulating weights via checkpoint."""
    similarity = AcousticSimilarity(sr=sr)
    producer = AcousticProducer(sr=sr, hidden_dim=32)
    baseline = 0.0

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading producer checkpoint: {checkpoint_path}")
        producer.load_checkpoint(checkpoint_path)

    # Load pre-trained substrate weights if provided
    if substrate_checkpoint_path and Path(substrate_checkpoint_path).exists():
        print(f"  Loading pre-trained substrate: {substrate_checkpoint_path}")
        import json as _json
        sub_data = _json.loads(Path(substrate_checkpoint_path).read_text())
        producer.substrate._W_h = np.array(sub_data["W_h"], dtype=np.float32)
        producer.substrate._W_x = np.array(sub_data["W_x"], dtype=np.float32)
        producer.substrate._b = np.array(sub_data["b"], dtype=np.float32)

    freq_results = {}

    for freq_idx, freq_hz in enumerate(freqs):
        print(f"\n--- Frequency {freq_idx + 1}/{len(freqs)}: {freq_hz:.1f} Hz ---")

        stimulus_pcm = generate_stimulus(freq_hz, duration_s, stimulus_waveform, sr)
        print(f"  Playing stimulus ({freq_hz:.1f} Hz)...")
        stimulus_result = play_and_capture(
            stimulus_pcm, input_device=input_device, output_device=stimulus_device, sr=sr,
        )
        stimulus_capture = stimulus_result["captured_pcm"]
        self_score = similarity.detailed_score(stimulus_pcm, stimulus_capture)
        print(f"  Self-score: {self_score.overall:.4f}")

        trial_rows = []
        for trial_idx in range(trials_per_freq):
            imitation_pcm = producer.produce(duration_s)
            imitation_result = play_and_capture(
                imitation_pcm, input_device=input_device, output_device=imitation_device, sr=sr,
            )
            imitation_capture = imitation_result["captured_pcm"]

            score = similarity.detailed_score(stimulus_capture, imitation_capture)
            reward = float(score.overall)
            baseline = 0.9 * baseline + 0.1 * reward
            reinforce_step(producer.controller, producer.substrate.hidden_state, reward, baseline, lr=lr)

            trial_rows.append({"trial": trial_idx + 1, "reward": reward, "baseline": float(baseline)})

            if (trial_idx + 1) % 10 == 0:
                print(f"  Trial {trial_idx + 1:3d}/{trials_per_freq}  reward={reward:.4f}")

        best = max(t["reward"] for t in trial_rows)
        imp = trial_rows[-1]["reward"] - trial_rows[0]["reward"]
        print(f"  Result: first={trial_rows[0]['reward']:.4f}  last={trial_rows[-1]['reward']:.4f}  "
              f"best={best:.4f}  improvement={imp:+.4f}")

        freq_results[f"{freq_hz:.0f}hz"] = {
            "first_reward": trial_rows[0]["reward"],
            "last_reward": trial_rows[-1]["reward"],
            "best_reward": best,
            "improvement": imp,
            "self_score": float(self_score.overall),
        }

        if checkpoint_path:
            producer.save_checkpoint(checkpoint_path)

    # Final evaluation on all frequencies
    print(f"\n--- Final Evaluation ---")
    eval_results = {}
    for freq_hz in freqs:
        stimulus_pcm = generate_stimulus(freq_hz, duration_s, stimulus_waveform, sr)
        stimulus_result = play_and_capture(
            stimulus_pcm, input_device=input_device, output_device=stimulus_device, sr=sr,
        )
        stimulus_capture = stimulus_result["captured_pcm"]

        imitation_pcm = producer.produce(duration_s)
        imitation_result = play_and_capture(
            imitation_pcm, input_device=input_device, output_device=imitation_device, sr=sr,
        )
        imitation_capture = imitation_result["captured_pcm"]
        score = similarity.detailed_score(stimulus_capture, imitation_capture)

        eval_results[f"{freq_hz:.0f}hz"] = {"overall": float(score.overall)}
        print(f"  {freq_hz:.0f} Hz: {score.overall:.4f}")

    return {
        "mode": "curriculum",
        "config": {"freqs": freqs, "trials_per_freq": trials_per_freq, "lr": lr},
        "freq_results": freq_results,
        "eval_results": eval_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 acoustic imitation loop")
    parser.add_argument("--stimulus-freq", type=float, default=None)
    parser.add_argument("--stimulus-waveform", type=str, default="sine",
                        choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Comma-separated frequencies: 220,262,330,392,440")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--stimulus-device", type=int, default=3)
    parser.add_argument("--imitation-device", type=int, default=2)
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Producer checkpoint (controller + substrate)")
    parser.add_argument("--substrate-checkpoint", type=str, default=None,
                        help="Pre-trained substrate checkpoint (from m9_substrate_pretrain.py)")
    parser.add_argument("--output", type=str, default="/tmp/m9_acoustic_imitation.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Acoustic Imitation Loop")
    print("=" * 72)
    print(f"  Stimulus device: {args.stimulus_device} (speaker)")
    print(f"  Imitation device: {args.imitation_device} (headphones)")
    print(f"  Input device: {args.input_device} (mic)")
    print(f"  Trials: {args.trials}")
    print(f"  Learning rate: {args.lr}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print()

    if args.curriculum:
        freqs = parse_freqs(args.curriculum)
        print(f"  Curriculum: {freqs}")
        result = run_curriculum(
            freqs=freqs,
            stimulus_waveform=args.stimulus_waveform,
            stimulus_device=args.stimulus_device,
            imitation_device=args.imitation_device,
            input_device=args.input_device,
            duration_s=args.duration,
            trials_per_freq=args.trials,
            lr=args.lr,
            checkpoint_path=args.checkpoint,
            substrate_checkpoint_path=args.substrate_checkpoint,
        )
    elif args.stimulus_freq:
        print(f"  Single frequency: {args.stimulus_freq:.1f} Hz {args.stimulus_waveform}")
        result = run_single_freq(
            stimulus_freq_hz=args.stimulus_freq,
            stimulus_waveform=args.stimulus_waveform,
            stimulus_device=args.stimulus_device,
            imitation_device=args.imitation_device,
            input_device=args.input_device,
            duration_s=args.duration,
            n_trials=args.trials,
            lr=args.lr,
            checkpoint_path=args.checkpoint,
            substrate_checkpoint_path=args.substrate_checkpoint,
        )
    else:
        parser.error("Specify --stimulus-freq or --curriculum")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
