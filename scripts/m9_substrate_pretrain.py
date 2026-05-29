#!/usr/bin/env python3
"""M9 Substrate Pre-Training — predictive audio encoder.

Trains the substrate (RNN) to predict the next audio frame from the current one.
This forces the hidden state to capture meaningful temporal/spectral features.

After pre-training, the substrate hidden state becomes a useful representation
that the controller can use to know what to imitate.
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

from m9_sound import AcousticProducer, AcousticSubstrate, NeuralController, OscillatorBank

SR = 16000


def generate_training_audio(
    n_seconds: float = 10.0,
    sr: int = SR,
) -> list[np.ndarray]:
    """Generate diverse audio clips for training."""
    osc = OscillatorBank(sr=sr)
    clips = []
    duration = n_seconds

    # Pure tones at various frequencies
    for freq in [110, 220, 330, 440, 550, 660, 880]:
        clips.append(osc.sine(freq, duration, amplitude=0.5))
        clips.append(osc.triangle(freq, duration, amplitude=0.4))
        clips.append(osc.sawtooth(freq, duration, amplitude=0.3))

    # Frequency sweeps
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    for start_freq in [100, 200, 400]:
        for end_freq in [400, 800, 1600]:
            if start_freq != end_freq:
                freqs = np.linspace(start_freq, end_freq, len(t)).astype(np.float32)
                sweep = (0.5 * np.sin(2 * np.pi * np.cumsum(freqs) / sr)).astype(np.float32)
                clips.append(sweep)

    return clips


def train_substrate_predictor(
    *,
    n_epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 32,
    frame_size: int = 480,
    sr: int = SR,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Train substrate to predict next audio frame."""

    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=sr, frame_size=frame_size)

    # Prediction head: hidden state → predicted next frame
    W_pred = np.random.randn(frame_size, hidden_dim).astype(np.float32) * 0.01
    b_pred = np.zeros(frame_size, dtype=np.float32)

    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        data = json.loads(Path(checkpoint_path).read_text())
        substrate._W_h = np.array(data["W_h"], dtype=np.float32)
        substrate._W_x = np.array(data["W_x"], dtype=np.float32)
        substrate._b = np.array(data["b"], dtype=np.float32)
        W_pred = np.array(data["W_pred"], dtype=np.float32)
        b_pred = np.array(data["b_pred"], dtype=np.float32)
        start_epoch = data.get("epoch", 0)

    # Generate training data
    print(f"Generating training audio...")
    clips = generate_training_audio(n_seconds=10.0, sr=sr)
    print(f"  {len(clips)} clips, {sum(len(c) for c in clips) / sr:.0f}s total audio")

    history = []

    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_steps = 0

        for clip in clips:
            substrate.reset()
            # Process frames, predict next
            for i in range(0, len(clip) - frame_size, frame_size):
                current_frame = clip[i:i + frame_size]
                next_frame = clip[i + frame_size:i + 2 * frame_size]

                if len(next_frame) < frame_size:
                    break

                # Forward: get hidden state from current frame
                h = substrate.step(current_frame)

                # Predict next frame
                predicted = W_pred @ h + b_pred

                # Loss: MSE between predicted and actual next frame
                error = predicted - next_frame
                loss = float(np.mean(error ** 2))
                epoch_loss += loss
                n_steps += 1

                # Backprop through prediction head only (keep RNN weights fixed for now)
                # d(L)/d(W_pred) = error @ h.T
                # d(L)/d(b_pred) = error
                dW_pred = np.outer(error, h)
                db_pred = error

                # Gradient clipping
                max_grad = 1.0
                grad_norm = np.linalg.norm(dW_pred)
                if grad_norm > max_grad:
                    dW_pred *= max_grad / grad_norm

                W_pred -= lr * dW_pred
                b_pred -= lr * db_pred

                # Also update RNN weights (backprop through one step)
                # d(L)/d(h) = W_pred.T @ error
                dh = W_pred.T @ error

                # Backprop through tanh: d(tanh)/dz = 1 - tanh^2
                # h = tanh(z), z = W_h @ h_prev + W_x @ x + b
                x = np.zeros(frame_size, dtype=np.float32)
                n = min(len(current_frame), frame_size)
                x[:n] = current_frame[:n]
                rms = np.sqrt(np.mean(x ** 2))
                if rms > 1e-6:
                    x = x / rms

                z = substrate._W_h @ substrate._h + substrate._W_x @ x + substrate._b
                dtanh = 1.0 - np.tanh(z) ** 2
                dz = dh * dtanh

                # Gradients for W_h, W_x, b
                # Note: we use h before this step, need to track it
                # For simplicity, skip W_h update (truncated BPTT)
                dW_x = np.outer(dz, x)
                db_rnn = dz

                # Clip RNN gradients
                rnn_grad_norm = np.linalg.norm(dW_x)
                if rnn_grad_norm > max_grad:
                    dW_x *= max_grad / rnn_grad_norm

                substrate._W_x -= lr * 0.1 * dW_x  # Slower RNN learning rate
                substrate._b -= lr * 0.1 * db_rnn

        avg_loss = epoch_loss / max(n_steps, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}  loss={avg_loss:.6f}")

    # Save checkpoint
    if checkpoint_path:
        data = {
            "W_h": substrate._W_h.tolist(),
            "W_x": substrate._W_x.tolist(),
            "b": substrate._b.tolist(),
            "W_pred": W_pred.tolist(),
            "b_pred": b_pred.tolist(),
            "config": {"hidden_dim": hidden_dim, "frame_size": frame_size, "sr": sr},
            "epoch": n_epochs,
            "final_loss": history[-1]["loss"] if history else 0,
        }
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(checkpoint_path).write_text(json.dumps(data))
        print(f"  Saved checkpoint: {checkpoint_path}")

    return {
        "history": history,
        "final_loss": history[-1]["loss"] if history else 0,
    }


def evaluate_substrate(
    substrate: AcousticSubstrate,
    freq_hz: float = 220.0,
    duration_s: float = 1.0,
    sr: int = SR,
) -> dict[str, Any]:
    """Evaluate: how different are hidden states for different frequencies?"""
    osc = OscillatorBank(sr=sr)
    substrate.reset()

    # Process a test tone
    test_tone = osc.sine(freq_hz, duration_s)
    n_frames = len(test_tone) // substrate._frame_size
    for i in range(n_frames):
        frame = test_tone[i * substrate._frame_size:(i + 1) * substrate._frame_size]
        substrate.step(frame)
    h_sine = substrate.hidden_state.copy()

    # Process a different frequency
    substrate.reset()
    test_tone2 = osc.sine(freq_hz * 1.5, duration_s)
    for i in range(n_frames):
        frame = test_tone2[i * substrate._frame_size:(i + 1) * substrate._frame_size]
        substrate.step(frame)
    h_diff = substrate.hidden_state.copy()

    # Process same frequency again (consistency check)
    substrate.reset()
    test_tone3 = osc.sine(freq_hz, duration_s)
    for i in range(n_frames):
        frame = test_tone3[i * substrate._frame_size:(i + 1) * substrate._frame_size]
        substrate.step(frame)
    h_repeat = substrate.hidden_state.copy()

    return {
        "same_freq_consistency": float(np.linalg.norm(h_sine - h_repeat)),
        "diff_freq_distance": float(np.linalg.norm(h_sine - h_diff)),
        "encoding_useful": float(np.linalg.norm(h_sine - h_diff)) > float(np.linalg.norm(h_sine - h_repeat)) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 substrate pre-training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="data/m9_substrate_pretrained.ckpt.json")
    parser.add_argument("--output", type=str, default="/tmp/m9_substrate_pretrain.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Substrate Pre-Training — Predictive Audio Encoder")
    print("=" * 72)
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()

    result = train_substrate_predictor(
        n_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        checkpoint_path=args.checkpoint,
    )

    # Evaluate encoding quality
    print("\nEvaluating substrate encoding...")
    substrate = AcousticSubstrate(hidden_dim=args.hidden_dim, sr=SR)
    data = json.loads(Path(args.checkpoint).read_text())
    substrate._W_h = np.array(data["W_h"], dtype=np.float32)
    substrate._W_x = np.array(data["W_x"], dtype=np.float32)
    substrate._b = np.array(data["b"], dtype=np.float32)

    eval_220 = evaluate_substrate(substrate, freq_hz=220.0)
    eval_440 = evaluate_substrate(substrate, freq_hz=440.0)

    print(f"  220 Hz encoding consistency: {eval_220['same_freq_consistency']:.4f}")
    print(f"  220 Hz vs 330 Hz distance: {eval_220['diff_freq_distance']:.4f}")
    print(f"  Encoding is useful: {eval_220['encoding_useful']}")
    print(f"  440 Hz encoding consistency: {eval_440['same_freq_consistency']:.4f}")
    print(f"  440 Hz vs 660 Hz distance: {eval_440['diff_freq_distance']:.4f}")

    result["eval"] = {"220hz": eval_220, "440hz": eval_440}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
