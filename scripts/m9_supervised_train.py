#!/usr/bin/env python3
"""M9 Supervised Controller Training.

Trains the controller to map substrate hidden states → oscillator params
using supervised learning (MSE loss on known target params).

This is much more efficient than REINFORCE because:
1. We know the exact target params for each stimulus
2. gradients flow directly through the network (no credit assignment problem)
3. Each sample provides a full gradient signal (not just a scalar reward)

After supervised pre-training, the controller should be able to produce
different frequencies for different stimuli. Then we can fine-tune with
REINFORCE for the full imitation loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticProducer, AcousticSimilarity, OscillatorBank, AcousticSubstrate
from m9_sound.controller import NeuralController

SR = 16000


def frequency_to_params(
    freq_hz: float,
    amplitude: float = 0.5,
    phase: float = 0.0,
    waveform: int = 0,
    n_waveforms: int = 4,
    freq_min: float = 80.0,
    freq_max: float = 2000.0,
) -> np.ndarray:
    """Convert known stimulus to target controller output params.

    Returns array: [freq_norm, amplitude, phase_norm, wf_onehot...]
    """
    freq_norm = (freq_hz - freq_min) / (freq_max - freq_min)
    freq_norm = np.clip(freq_norm, 0.0, 1.0)
    phase_norm = phase / np.pi  # normalize to [-1, 1]

    target = np.zeros(4 + n_waveforms, dtype=np.float32)
    target[0] = freq_norm
    target[1] = amplitude
    target[2] = phase_norm
    target[3 + waveform] = 1.0  # one-hot waveform

    return target


def train_controller_supervised(
    *,
    freqs: list[float],
    amplitude: float = 0.5,
    waveform: int = 0,
    substrate_checkpoint: str,
    n_repeats: int = 10,
    lr: float = 0.01,
    n_epochs: int = 100,
    output_checkpoint: str | None = None,
    sr: int = SR,
    hidden_dim: int = 32,
    controller_hidden_dim: int = 32,
) -> dict[str, Any]:
    """Train controller via supervised learning.

    For each frequency:
    1. Generate stimulus, encode through pre-trained substrate to get hidden state
    2. Compute target params from known stimulus
    3. Train controller: hidden_state to target_params (MSE loss)
    """

    # Load pre-trained substrate
    sub_data = json.loads(Path(substrate_checkpoint).read_text())
    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=sr)
    substrate._W_h = np.array(sub_data["W_h"], dtype=np.float32)
    substrate._W_x = np.array(sub_data["W_x"], dtype=np.float32)
    substrate._b = np.array(sub_data["b"], dtype=np.float32)
    print(f"Loaded pre-trained substrate from {substrate_checkpoint}")

    # Create a fresh controller
    controller = NeuralController(hidden_dim=hidden_dim, sr=sr, controller_hidden_dim=controller_hidden_dim)
    print(f"Controller: {controller.n_params} params")

    osc = OscillatorBank(sr=sr)

    # Collect training data: hidden states and target params
    all_hidden = []
    all_targets = []

    print(f"\nGenerating training data for {len(freqs)} frequencies...")
    for freq_hz in freqs:
        target_params = frequency_to_params(
            freq_hz, amplitude=amplitude, waveform=waveform
        )

        for repeat in range(n_repeats):
            # Generate stimulus
            stimulus = osc.sine(freq_hz, 1.0, amplitude=amplitude)

            # Add small random phase offset for diversity
            phase_offset = np.random.uniform(0, 2 * np.pi)
            stimulus = osc.sine(freq_hz, 1.0, amplitude=amplitude, phase=phase_offset)

            # Encode through substrate
            substrate.reset()
            frame_size = 480
            n_frames = len(stimulus) // frame_size
            for i in range(n_frames):
                frame = stimulus[i * frame_size:(i + 1) * frame_size]
                substrate.step(frame)

            h = substrate.hidden_state.copy()
            all_hidden.append(h)
            all_targets.append(target_params)

    hidden_states = np.array(all_hidden, dtype=np.float32)
    target_params = np.array(all_targets, dtype=np.float32)

    print(f"Training data: {hidden_states.shape[0]} samples")
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Target params: {target_params.shape}")

    # Check: are hidden states actually different for different freqs?
    print(f"\nHidden state analysis:")
    for i, freq in enumerate(freqs):
        mask = np.arange(len(freqs) * n_repeats) % len(freqs) == i
        h_mean = hidden_states[mask].mean(axis=0)
        print(f"  {freq} Hz: mean_norm={np.linalg.norm(h_mean):.4f}")

    # Compute target params with correct frequency normalization
    all_targets_corrected = []
    for freq_hz in freqs:
        for repeat in range(n_repeats):
            target = frequency_to_params(freq_hz, amplitude=amplitude, waveform=waveform)
            all_targets_corrected.append(target)
    target_params = np.array(all_targets_corrected, dtype=np.float32)

    # Train controller
    print(f"\nTraining controller (lr={lr}, epochs={n_epochs})...")
    history = controller.train_supervised(
        hidden_states, target_params, lr=lr, n_epochs=n_epochs
    )

    # Evaluate
    print(f"\nEvaluation:")
    sim = AcousticSimilarity(sr=sr)
    for freq in freqs:
        # Generate stimulus and encode
        stimulus = osc.sine(freq, 1.0)
        substrate.reset()
        frame_size = 480
        for i in range(len(stimulus) // frame_size):
            frame = stimulus[i * frame_size:(i + 1) * frame_size]
            substrate.step(frame)
        h = substrate.hidden_state

        # Controller predicts params
        params = controller.decode(h)

        # Generate imitation using predicted params
        osc2 = OscillatorBank(sr=sr)
        wf_map = [osc2.sine, osc2.triangle, osc2.sawtooth, osc2.square]
        imitation = wf_map[params.waveform](params.freq_hz, 1.0, amplitude=params.amplitude)

        # Compare
        target = osc.sine(freq, 1.0)
        score = sim.detailed_score(target, imitation)
        print(f"  {freq} Hz: produced {params.freq_hz:.1f} Hz, score={score.overall:.4f}")

    # Save checkpoint
    if output_checkpoint:
        data = {
            "controller": {
                "W1": controller._W1.tolist(),
                "b1": controller._b1.tolist(),
                "W2": controller._W2.tolist(),
                "b2": controller._b2.tolist(),
            },
            "config": {
                "hidden_dim": hidden_dim,
                "controller_hidden_dim": controller_hidden_dim,
                "sr": sr,
                "freqs": freqs,
                "n_samples": len(all_hidden),
                "final_loss": history[-1] if history else 0,
            },
        }
        Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        Path(output_checkpoint).write_text(json.dumps(data))
        print(f"\nSaved checkpoint: {output_checkpoint}")

    return {
        "history": history,
        "final_loss": history[-1] if history else 0,
        "freqs": freqs,
        "n_samples": len(all_hidden),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 supervised controller training")
    parser.add_argument("--freqs", type=str, default="220,262,330,392,440",
                        help="Comma-separated training frequencies")
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--waveform", type=int, default=0, help="0=sine,1=triangle,2=sawtooth,3=square")
    parser.add_argument("--substrate-checkpoint", type=str, required=True,
                        help="Pre-trained substrate checkpoint")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Samples per frequency")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--controller-hidden", type=int, default=32,
                        help="Controller hidden layer dimension")
    parser.add_argument("--output", type=str, default="/tmp/m9_supervised.json")
    parser.add_argument("--checkpoint", type=str, default="data/m9_controller_supervised.ckpt.json")
    args = parser.parse_args()

    freqs = [float(f.strip()) for f in args.freqs.split(",")]

    print("=" * 72)
    print("M9 Supervised Controller Training")
    print("=" * 72)
    print(f"  Frequencies: {freqs}")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Waveform: {args.waveform}")
    print(f"  Repeats per freq: {args.repeats}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Substrate checkpoint: {args.substrate_checkpoint}")
    print(f"  Output checkpoint: {args.checkpoint}")
    print()

    result = train_controller_supervised(
        freqs=freqs,
        amplitude=args.amplitude,
        waveform=args.waveform,
        substrate_checkpoint=args.substrate_checkpoint,
        n_repeats=args.repeats,
        lr=args.lr,
        n_epochs=args.epochs,
        output_checkpoint=args.checkpoint,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
