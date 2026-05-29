#!/usr/bin/env python3
"""M9 Substrate Pre-Training — frequency-discriminating audio encoder.

Trains the substrate (RNN) with a frequency classification objective.
This forces the hidden state to encode frequency distinctly, which is
essential for the controller to distinguish between different stimuli.

Key insight from previous attempts: next-frame prediction alone causes
the substrate to saturate (tanh -> all ±1) and all frequencies produce
nearly identical hidden states. Classification forces discrimination.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticSubstrate, OscillatorBank

SR = 16000


def generate_training_data(
    n_seconds: float = 5.0,
    sr: int = SR,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Generate labeled audio clips: (clip, frequency_class)."""
    osc = OscillatorBank(sr=sr)
    clips = []
    labels = []

    freq_bins = [110, 220, 262, 330, 392, 440, 550, 660, 880, 1100]
    duration = n_seconds

    for freq_idx, freq in enumerate(freq_bins):
        # Pure sine
        clips.append(osc.sine(freq, duration, amplitude=0.5))
        labels.append(freq_idx)
        # Phase-shifted variants for diversity
        for phase in [0.5, 1.0, 1.5, 2.0]:
            clips.append(osc.sine(freq, duration, amplitude=0.5, phase=phase))
            labels.append(freq_idx)
        # Triangle
        clips.append(osc.triangle(freq, duration, amplitude=0.4))
        labels.append(freq_idx)
        # Sawtooth
        clips.append(osc.sawtooth(freq, duration, amplitude=0.3))
        labels.append(freq_idx)
        # Mixed: two frequencies
        mixed = 0.3 * osc.sine(freq, duration) + 0.2 * osc.sine(freq * 1.5, duration)
        clips.append(mixed)
        labels.append(freq_idx)

    return clips, labels, freq_bins


def train_substrate_classifier(
    *,
    n_epochs: int = 100,
    lr: float = 0.001,
    hidden_dim: int = 32,
    frame_size: int = 480,
    sr: int = SR,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Train substrate for frequency classification."""

    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=sr, frame_size=frame_size)

    # Classification head: hidden_state → frequency class logits
    n_classes = 10  # number of frequency bins
    W_cls = np.random.randn(n_classes, hidden_dim).astype(np.float32) * 0.1
    b_cls = np.zeros(n_classes, dtype=np.float32)

    # Load checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        data = json.loads(Path(checkpoint_path).read_text())
        substrate._W_h = np.array(data["W_h"], dtype=np.float32)
        substrate._W_x = np.array(data["W_x"], dtype=np.float32)
        substrate._b = np.array(data["b"], dtype=np.float32)
        W_cls = np.array(data["W_cls"], dtype=np.float32)
        b_cls = np.array(data["b_cls"], dtype=np.float32)

    # Generate training data
    print(f"Generating training data...")
    clips, labels, freq_bins = generate_training_data(n_seconds=5.0, sr=sr)
    print(f"  {len(clips)} clips, {len(freq_bins)} frequency classes")

    history = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Shuffle training data
        indices = np.random.permutation(len(clips))

        for idx in indices:
            clip = clips[idx]
            label = labels[idx]

            # Process full clip through substrate
            substrate.reset()
            n_frames = len(clip) // frame_size
            for i in range(n_frames):
                frame = clip[i * frame_size:(i + 1) * frame_size]
                substrate.step(frame)

            h = substrate.hidden_state

            # Classify: softmax cross-entropy
            logits = W_cls @ h + b_cls
            # Numerically stable softmax
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum()

            # Cross-entropy loss
            loss = -np.log(probs[label] + 1e-10)
            total_loss += loss

            # Accuracy
            pred = np.argmax(probs)
            if pred == label:
                correct += 1
            total += 1

            # Backprop through classification head
            # d(cross_entropy)/d(logits) = probs - one_hot
            dlogits = probs.copy()
            dlogits[label] -= 1.0

            dW_cls = np.outer(dlogits, h)
            db_cls = dlogits

            # Backprop through substrate (single step BPTT)
            dh = W_cls.T @ dlogits

            # Get the last input frame for gradient computation
            last_frame = clip[(n_frames - 1) * frame_size:n_frames * frame_size]
            x = np.zeros(frame_size, dtype=np.float32)
            n = min(len(last_frame), frame_size)
            x[:n] = last_frame[:n]
            rms = np.sqrt(np.mean(x ** 2))
            if rms > 1e-6:
                x = x / rms

            # Backprop through tanh
            z = substrate._W_h @ substrate._h + substrate._W_x @ x + substrate._b
            dz = dh * (1.0 - np.tanh(z) ** 2)

            dW_x = np.outer(dz, x)
            db_rnn = dz

            # Gradient clipping
            max_grad = 5.0
            for g in [dW_cls, db_cls, dW_x, db_rnn]:
                gn = np.linalg.norm(g)
                if gn > max_grad:
                    g *= max_grad / gn

            # Update weights
            W_cls -= lr * dW_cls
            b_cls -= lr * db_cls
            substrate._W_x -= lr * 0.1 * dW_x
            substrate._b -= lr * 0.1 * db_rnn

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        history.append({"epoch": epoch + 1, "loss": float(avg_loss), "accuracy": float(accuracy)})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={accuracy:.3f}")

    # Evaluate: check hidden state discriminability
    print(f"\nEvaluating hidden state discriminability...")
    hidden_by_class = {i: [] for i in range(len(freq_bins))}
    for clip, label in zip(clips[:30], labels[:30]):  # subset for speed
        substrate.reset()
        for i in range(len(clip) // frame_size):
            frame = clip[i * frame_size:(i + 1) * frame_size]
            substrate.step(frame)
        hidden_by_class[label].append(substrate.hidden_state.copy())

    print(f"  Mean hidden state norms per class:")
    for cls_idx in range(len(freq_bins)):
        if hidden_by_class[cls_idx]:
            states = np.array(hidden_by_class[cls_idx])
            mean_norm = np.linalg.norm(states.mean(axis=0))
            print(f"    {freq_bins[cls_idx]:4d} Hz: norm={mean_norm:.4f}")

    # Between-class distances
    print(f"\n  Between-class distances:")
    for i in range(len(freq_bins)):
        for j in range(i + 1, len(freq_bins)):
            if hidden_by_class[i] and hidden_by_class[j]:
                states_i = np.array(hidden_by_class[i]).mean(axis=0)
                states_j = np.array(hidden_by_class[j]).mean(axis=0)
                dist = np.linalg.norm(states_i - states_j)
                if dist < 2.0:  # Only print close ones
                    print(f"    {freq_bins[i]} vs {freq_bins[j]}: {dist:.4f}")

    # Save checkpoint
    if checkpoint_path:
        data = {
            "W_h": substrate._W_h.tolist(),
            "W_x": substrate._W_x.tolist(),
            "b": substrate._b.tolist(),
            "W_cls": W_cls.tolist(),
            "b_cls": b_cls.tolist(),
            "config": {"hidden_dim": hidden_dim, "frame_size": frame_size, "sr": sr},
            "epoch": n_epochs,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "freq_bins": freq_bins,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(checkpoint_path).write_text(json.dumps(data))
        print(f"\nSaved checkpoint: {checkpoint_path}")

    return {"history": history, "freq_bins": freq_bins, "final_accuracy": history[-1]["accuracy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 substrate pre-training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="data/m9_substrate_classifier.ckpt.json")
    parser.add_argument("--output", type=str, default="/tmp/m9_substrate_pretrain.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Substrate Pre-Training — Frequency Classification")
    print("=" * 72)
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print()

    result = train_substrate_classifier(
        n_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        checkpoint_path=args.checkpoint,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
