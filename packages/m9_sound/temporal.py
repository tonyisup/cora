"""Helpers for M9 temporal-sequence smoke testing.

This module factors the small sequence-demo logic out of the ad-hoc scripts so
we can regression-test the temporal path without any audio hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from m9_sound.controller import ControllerParams
from m9_sound.producer import AcousticProducer

SR = 16000
FRAME_SIZE = 480
DEFAULT_TEMPORAL_TONES: list[tuple[float, float]] = [
    (220.0, 0.12),
    (330.0, 0.12),
    (440.0, 0.12),
]

TEMPORAL_FAMILIES: dict[str, list[tuple[float, float]]] = {
    "default": DEFAULT_TEMPORAL_TONES,
    "diphthong": [
        (220.0, 0.06),
        (330.0, 0.06),
        (440.0, 0.06),
        (330.0, 0.06),
        (220.0, 0.06),
    ],
    "wide_sweep": [
        (220.0, 0.09),
        (330.0, 0.09),
        (440.0, 0.09),
        (550.0, 0.09),
        (660.0, 0.09),
        (550.0, 0.09),
        (440.0, 0.09),
    ],
}

DEVICE_PROFILES = ("flat", "highpass")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROLLER_CKPT = REPO_ROOT / "data" / "m9_controller_temporal_v3.ckpt.json"
DEFAULT_SUBSTRATE_CKPT = REPO_ROOT / "data" / "m9_substrate_live.ckpt.json"


def generate_tone_sequence(
    tones: Sequence[tuple[float, float]],
    *,
    sr: int = SR,
    amplitude: float = 0.5,
    phase_offset: float = 0.0,
    noise_rms: float = 0.0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Generate a deterministic tone sequence for temporal smoke tests."""
    parts: list[np.ndarray] = []
    current_phase = float(phase_offset)
    rng = np.random.default_rng(noise_seed) if noise_rms > 0.0 else None

    for freq_hz, duration_s in tones:
        n_samples = int(round(sr * duration_s))
        if n_samples <= 0:
            continue
        t = np.arange(n_samples, dtype=np.float32) / sr
        frame = amplitude * np.sin(2.0 * np.pi * float(freq_hz) * t + current_phase)
        if rng is not None:
            frame = frame + rng.normal(0.0, noise_rms, size=n_samples).astype(np.float32)
        parts.append(frame.astype(np.float32))
        current_phase = float(
            (current_phase + 2.0 * np.pi * float(freq_hz) * (n_samples / sr)) % (2.0 * np.pi)
        )

    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def frame_audio(audio: np.ndarray, *, frame_size: int = FRAME_SIZE) -> list[np.ndarray]:
    """Split audio into full frames, dropping any trailing partial frame."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_frames = len(audio) // frame_size
    return [audio[i * frame_size : (i + 1) * frame_size].copy() for i in range(n_frames)]


def apply_device_profile(
    audio: np.ndarray,
    *,
    device_profile: str = "flat",
    device_gain: float = 1.0,
) -> np.ndarray:
    """Apply a deterministic device profile used by the smoke regression."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1) * float(device_gain)
    if device_profile == "flat":
        return audio.astype(np.float32)
    if device_profile == "highpass":
        # Simple first-order high-pass to mimic the common capture-device profile.
        y = np.empty_like(audio)
        prev_x = 0.0
        prev_y = 0.0
        alpha = 0.98
        for idx, sample in enumerate(audio):
            y[idx] = alpha * (prev_y + sample - prev_x)
            prev_x = float(sample)
            prev_y = float(y[idx])
        return y.astype(np.float32)
    raise ValueError(f"Unknown device profile: {device_profile}")


def _render_frame(params: ControllerParams, *, sr: int, frame_size: int, cumulative_phase: float) -> np.ndarray:
    t = np.arange(frame_size, dtype=np.float32) / sr
    phase_offset = cumulative_phase + float(params.phase)

    if params.waveform == 0:
        frame = params.amplitude * np.sin(2.0 * np.pi * params.freq_hz * t + phase_offset)
    elif params.waveform == 1:
        ramp = 2.0 * ((params.freq_hz * t + phase_offset / (2.0 * np.pi)) % 1.0) - 1.0
        frame = params.amplitude * (2.0 * np.abs(ramp) - 1.0)
    elif params.waveform == 2:
        frame = params.amplitude * (
            2.0 * ((params.freq_hz * t + phase_offset / (2.0 * np.pi)) % 1.0) - 1.0
        )
    else:
        frame = params.amplitude * np.sign(
            np.sin(2.0 * np.pi * params.freq_hz * t + phase_offset)
        )

    return frame.astype(np.float32)


def render_param_sequence(
    params: Sequence[ControllerParams],
    *,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    """Render controller parameters into audio and return a continuity report."""
    if not params:
        report = {
            "frame_count": 0,
            "max_boundary_delta": 0.0,
            "min_boundary_delta": 0.0,
            "mean_boundary_delta": 0.0,
            "freq_min": 0.0,
            "freq_max": 0.0,
        }
        return np.zeros(0, dtype=np.float32), report

    frames: list[np.ndarray] = []
    boundary_deltas: list[float] = []
    cumulative_phase = 0.0

    for idx, param in enumerate(params):
        frame = _render_frame(param, sr=sr, frame_size=frame_size, cumulative_phase=cumulative_phase)
        if frames:
            boundary_deltas.append(float(abs(frame[0] - frames[-1][-1])))
        frames.append(frame)
        cumulative_phase = float(
            (cumulative_phase + 2.0 * np.pi * float(param.freq_hz) * (frame_size / sr))
            % (2.0 * np.pi)
        )

    audio = np.concatenate(frames).astype(np.float32)
    freq_hz = [float(p.freq_hz) for p in params]
    report = {
        "frame_count": len(params),
        "max_boundary_delta": float(max(boundary_deltas) if boundary_deltas else 0.0),
        "min_boundary_delta": float(min(boundary_deltas) if boundary_deltas else 0.0),
        "mean_boundary_delta": float(np.mean(boundary_deltas) if boundary_deltas else 0.0),
        "freq_min": float(min(freq_hz)),
        "freq_max": float(max(freq_hz)),
        "freq_span": float(max(freq_hz) - min(freq_hz)),
    }
    return audio, report


def load_temporal_demo_producer(
    *,
    controller_ckpt_path: str | Path = DEFAULT_CONTROLLER_CKPT,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> AcousticProducer:
    """Load the saved temporal demo checkpoints into an AcousticProducer."""
    controller_ckpt_path = Path(controller_ckpt_path)
    substrate_ckpt_path = Path(substrate_ckpt_path)

    if not controller_ckpt_path.exists():
        raise FileNotFoundError(f"Missing controller checkpoint: {controller_ckpt_path}")
    if not substrate_ckpt_path.exists():
        raise FileNotFoundError(f"Missing substrate checkpoint: {substrate_ckpt_path}")

    producer = AcousticProducer(sr=sr, hidden_dim=32, frame_size=frame_size)

    controller_ckpt = json.loads(controller_ckpt_path.read_text())
    producer.controller._W1 = np.array(controller_ckpt["W1"], dtype=np.float32)
    producer.controller._b1 = np.array(controller_ckpt["b1"], dtype=np.float32)
    producer.controller._W2 = np.array(controller_ckpt["W2"], dtype=np.float32)
    producer.controller._b2 = np.array(controller_ckpt["b2"], dtype=np.float32)

    substrate_ckpt = json.loads(substrate_ckpt_path.read_text())
    producer.substrate._W_h = np.array(substrate_ckpt["W_h"], dtype=np.float32)
    producer.substrate._W_x = np.array(substrate_ckpt["W_x"], dtype=np.float32)
    producer.substrate._b = np.array(substrate_ckpt["b"], dtype=np.float32)

    return producer


def decode_temporal_sequence(
    producer: AcousticProducer,
    stimulus: np.ndarray,
    *,
    frame_size: int = FRAME_SIZE,
) -> list[ControllerParams]:
    """Run stimulus frames through the temporal substrate/controller stack."""
    frames = frame_audio(stimulus, frame_size=frame_size)
    producer.reset()
    params: list[ControllerParams] = []

    for frame in frames:
        producer.substrate.step(frame)
        params.append(producer.controller.decode(producer.substrate.hidden_state))

    return params


def count_decoded_transitions(
    params: Sequence[ControllerParams],
    *,
    freq_threshold_hz: float = 40.0,
    smooth_window: int = 3,
) -> int:
    """Count coarse transitions in a decoded frequency trace.

    A small median filter makes the count robust to jitter in noisy captures.
    """
    if not params:
        return 0

    freqs = np.array([float(p.freq_hz) for p in params], dtype=np.float32)
    if smooth_window > 1 and len(freqs) >= smooth_window:
        half = smooth_window // 2
        smoothed = freqs.copy()
        for idx in range(len(freqs)):
            lo = max(0, idx - half)
            hi = min(len(freqs), idx + half + 1)
            smoothed[idx] = float(np.median(freqs[lo:hi]))
        freqs = smoothed

    transitions = 0
    prev = float(freqs[0])
    for freq in freqs[1:]:
        if abs(float(freq) - prev) >= freq_threshold_hz:
            transitions += 1
            prev = float(freq)
    return transitions


def run_temporal_smoke_report(
    *,
    family: str = "default",
    tones: Sequence[tuple[float, float]] | None = None,
    controller_ckpt_path: str | Path = DEFAULT_CONTROLLER_CKPT,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
    input_noise_rms: float = 0.0,
    noise_seed: int | None = None,
    device_profile: str = "flat",
    device_gain: float = 1.0,
) -> dict[str, object]:
    """Run a lightweight temporal-sequence smoke check and return a JSONable report."""
    producer = load_temporal_demo_producer(
        controller_ckpt_path=controller_ckpt_path,
        substrate_ckpt_path=substrate_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )

    if tones is None:
        if family not in TEMPORAL_FAMILIES:
            raise ValueError(f"Unknown temporal family: {family}")
        tones = TEMPORAL_FAMILIES[family]

    stimulus = generate_tone_sequence(
        tones,
        sr=sr,
        noise_rms=input_noise_rms,
        noise_seed=noise_seed,
    )
    stimulus = apply_device_profile(stimulus, device_profile=device_profile, device_gain=device_gain)
    params = decode_temporal_sequence(producer, stimulus, frame_size=frame_size)
    rendered, render_report = render_param_sequence(params, sr=sr, frame_size=frame_size)
    stimulus_frames = frame_audio(stimulus, frame_size=frame_size)

    decoded_transition_count = count_decoded_transitions(
        params,
        smooth_window=5 if device_profile != "flat" else 3,
    )

    return {
        "stimulus_tones": [[float(freq), float(duration)] for freq, duration in tones],
        "family": family,
        "checkpoint_paths": {
            "controller": str(Path(controller_ckpt_path)),
            "substrate": str(Path(substrate_ckpt_path)),
        },
        "stimulus_samples": int(len(stimulus)),
        "stimulus_frame_count": int(len(stimulus_frames)),
        "rendered_samples": int(len(rendered)),
        "rendered_frame_count": int(render_report["frame_count"]),
        "input_transition_count": int(max(0, len(tones) - 1)),
        "decoded_transition_count": int(decoded_transition_count),
        "decoded_freq_min": float(render_report["freq_min"]),
        "decoded_freq_max": float(render_report["freq_max"]),
        "decoded_freq_span": float(render_report["freq_span"]),
        "max_boundary_delta": float(render_report["max_boundary_delta"]),
        "min_boundary_delta": float(render_report["min_boundary_delta"]),
        "mean_boundary_delta": float(render_report["mean_boundary_delta"]),
        "input_noise_rms": float(input_noise_rms),
        "noise_seed": noise_seed,
        "device_profile": device_profile,
        "device_gain": float(device_gain),
    }
