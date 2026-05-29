#!/usr/bin/env python3
"""M9 Substrate Pre-Training on Live Audio.

Plays training stimuli through the speaker, captures them through the mic,
and trains the substrate to classify frequencies from real audio.

This solves the domain gap: synthetic audio produces different hidden
states than captured audio. Training on live captures makes the substrate
robust to room noise, distortion, and mic frequency response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import AcousticSubstrate, OscillatorBank

SR = 16000


def play_stimulus_and_capture(
    pcm: np.ndarray,
    input_device: int = 1,
    output_device: int = 2,
    sr: int = SR,
) -> np.ndarray:
    """Play audio on output_device and capture from input_device."""
    import sounddevice as sd

    if pcm.ndim != 1:
        pcm = pcm.reshape(-1)
    pcm = pcm.astype(np.float32)

    input_info = sd.query_devices(input_device)
    input_sr = int(input_info["default_samplerate"] or sr)

    lead_in = int(0.1 * input_sr)
    playback_duration = len(pcm) / sr
    total_duration = 0.1 + playback_duration + 0.1
    total_samples = int(total_duration * input_sr)
    captured = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    playback_started = False

    stream = sd.InputStream(
        device=input_device, channels=1, samplerate=input_sr,
        dtype="float32", blocksize=1024,
    )
    stream.start()
    try:
        while pos < total_samples:
            if not playback_started and pos >= lead_in:
                sd.play(pcm, samplerate=sr, device=output_device, blocking=False)
                playback_started = True
            remaining = total_samples - pos
            to_read = min(1024, remaining)
            data, _ = stream.read(to_read)
            n = min(len(data), remaining)
            captured[pos : pos + n] = data[:n, 0]
            pos += n
    finally:
        stream.stop()
        stream.close()
        sd.wait()

    # Resample to target SR if needed
    if input_sr != sr:
        from scipy.signal import resample_poly
        captured = resample_poly(captured, sr, input_sr).astype(np.float32)

    return captured


def generate_live_training_data(
    *,
    n_seconds: float = 3.0,
    sr: int = SR,
    input_device: int = 1,
    output_device: int = 2,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Play stimuli through speaker, capture through mic, return labeled data."""
    osc = OscillatorBank(sr=sr)
    clips = []
    labels = []

    freq_bins = [110, 220, 262, 330, 392, 440, 550, 660, 880]
    duration = n_seconds

    print(f"Generating live training data ({len(freq_bins)} frequencies)...")
    print(f"  Playing on device {output_device}, capturing on device {input_device}")

    for freq_idx, freq in enumerate(freq_bins):
        print(f"  Freq {freq_idx+1}/{len(freq_bins)}: {freq} Hz...", end="", flush=True)

        # Generate stimulus
        stimulus = osc.sine(freq, duration, amplitude=0.5)

        # Play and capture
        captured = play_stimulus_and_capture(
            stimulus, input_device=input_device, output_device=output_device, sr=sr
        )

        clips.append(captured)
        labels.append(freq_idx)

        # Verify capture quality
        rms = np.sqrt(np.mean(captured ** 2))
        print(f"  captured {len(captured)} samples, RMS={rms:.4f}")

    return clips, labels, freq_bins


def train_substrate_on_live_audio(
    *,
    n_epochs: int = 200,
    lr: float = 0.001,
    hidden_dim: int = 32,
    frame_size: int = 480,
    sr: int = SR,
    checkpoint_path: str | None = None,
    input_device: int = 1,
    output_device: int = 2,
) -> dict[str, Any]:
    """Train substrate classifier on live captured audio."""

    # Generate live training data
    clips, labels, freq_bins = generate_live_training_data(
        n_seconds=3.0, sr=sr, input_device=input_device, output_device=output_device
    )

    n_classes = len(freq_bins)

    # Initialize substrate and classification head
    substrate = AcousticSubstrate(hidden_dim=hidden_dim, sr=sr, frame_size=frame_size)
    W_cls = np.random.randn(n_classes, hidden_dim).astype(np.float32) * 0.1
    b_cls = np.zeros(n_classes, dtype=np.float32)

    # Load checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        data = json.loads(Path(checkpoint_path).read_text())
        if "W_h" in data:
            substrate._W_h = np.array(data["W_h"], dtype=np.float32)
            substrate._W_x = np.array(data["W_x"], dtype=np.float32)
            substrate._b = np.array(data["b"], dtype=np.float32)
        if "W_cls" in data:
            W_cls = np.array(data["W_cls"], dtype=np.float32)
            b_cls = np.array(data["b_cls"], dtype=np.float32)

    history = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        indices = np.random.permutation(len(clips))

        for idx in indices:
            clip = clips[idx]
            label = labels[idx]

            # Process full clip through substrate
            substrate.reset()
            n_frames = len(clip) // frame_size
            for i in range(n_frames):
                frame = clip[i * frame_size : (i + 1) * frame_size]
                substrate.step(frame)

            h = substrate.hidden_state

            # Classify
            logits = W_cls @ h + b_cls
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum()

            loss = -np.log(probs[label] + 1e-10)
            total_loss += loss

            pred = np.argmax(probs)
            if pred == label:
                correct += 1
            total += 1

            # Backprop
            dlogits = probs.copy()
            dlogits[label] -= 1.0

            dW_cls = np.outer(dlogits, h)
            db_cls = dlogits

            # Backprop to substrate (truncated BPTT through last frame)
            dh = W_cls.T @ dlogits
            last_frame = clip[(n_frames - 1) * frame_size : n_frames * frame_size]
            x = np.zeros(frame_size, dtype=np.float32)
            n = min(len(last_frame), frame_size)
            x[:n] = last_frame[:n]
            rms = np.sqrt(np.mean(x ** 2))
            if rms > 1e-6:
                x = x / rms

            z = substrate._W_h @ substrate._h + substrate._W_x @ x + substrate._b
            dz = dh * (1.0 - np.tanh(z) ** 2)

            dW_x = np.outer(dz, x)
            db_rnn = dz

            max_grad = 5.0
            for g in [dW_cls, db_cls, dW_x, db_rnn]:
                gn = np.linalg.norm(g)
                if gn > max_grad:
                    g *= max_grad / gn

            W_cls -= lr * dW_cls
            b_cls -= lr * db_cls
            substrate._W_x -= lr * 0.1 * dW_x
            substrate._b -= lr * 0.1 * db_rnn

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        history.append({"epoch": epoch + 1, "loss": float(avg_loss), "accuracy": float(accuracy)})

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={accuracy:.3f}")

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
        print(f"Saved checkpoint: {checkpoint_path}")

    return {"history": history, "freq_bins": freq_bins, "final_accuracy": history[-1]["accuracy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 substrate pre-training on live audio")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--input-device", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="data/m9_substrate_live.ckpt.json")
    parser.add_argument("--output", type=str, default="/tmp/m9_substrate_live.json")
    args = parser.parse_args()

    print("=" * 72)
    print("M9 Substrate Pre-Training — Live Audio")
    print("=" * 72)
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Input device: {args.input_device} (mic)")
    print(f"  Output device: {args.output_device} (speaker)")
    print()

    result = train_substrate_on_live_audio(
        n_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        checkpoint_path=args.checkpoint,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
