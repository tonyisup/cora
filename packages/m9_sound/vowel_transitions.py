"""M9 vowel-transition family helpers.

This module keeps the new M9 slice deterministic and reusable:
- generate small vowel-transition families from the spectral lookup data
- synthesize listenable audio with phase continuity
- run the saved spectral substrate over the rendered frames
- score primary vs held-out families and optionally write a WAV artifact
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from m9_sound.temporal import FRAME_SIZE, SR, apply_device_profile, frame_audio, write_pcm16_wav

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSTRATE_CKPT = REPO_ROOT / "data" / "m9_substrate_spectral.ckpt.json"
DEFAULT_LOOKUP_CKPT = REPO_ROOT / "data" / "m9_formant_lookup_v5.json"

VOWEL_FAMILIES: dict[str, list[str]] = {
    "default": ["i", "E", "u"],
    "diphthong": ["i", "E", "a", "O", "u"],
    "wide_sweep": ["i", "E", "a", "O", "u", "O", "a"],
}

FAMILY_DURATIONS_S: dict[str, float] = {
    "default": 0.12,
    "diphthong": 0.06,
    "wide_sweep": 0.09,
}

FORMANT_KEYS = (("f1", "a1"), ("f2", "a2"), ("f3", "a3"))


@lru_cache(maxsize=1)
def _load_lookup_data(lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT) -> Any:
    path = Path(lookup_ckpt_path)
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_substrate_weights(substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT) -> dict[str, np.ndarray | int]:
    path = Path(substrate_ckpt_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "W_h": np.asarray(raw["W_h"], dtype=np.float32),
        "W_x": np.asarray(raw["W_x"], dtype=np.float32),
        "b": np.asarray(raw["b"], dtype=np.float32),
        "hidden_dim": int(raw.get("hidden_dim", 32)),
        "n_mel": int(raw.get("n_mel", 26)),
        "sr": int(raw.get("sr", SR)),
    }


@lru_cache(maxsize=1)
def _load_centroids(lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT) -> tuple[list[str], np.ndarray]:
    lookup_data = _load_lookup_data(lookup_ckpt_path)
    vowel_names = list(lookup_data["vowel_names"])
    centroids = np.asarray(
        [lookup_data["lookup"][vowel]["centroid"] for vowel in vowel_names],
        dtype=np.float32,
    )
    return vowel_names, centroids


@lru_cache(maxsize=1)
def _load_templates(lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT) -> dict[str, dict[str, float]]:
    lookup_data = _load_lookup_data(lookup_ckpt_path)
    return {
        vowel: dict(lookup_data["lookup"][vowel]["template"])
        for vowel in lookup_data["vowel_names"]
    }


def _mel_filterbank(n_mels: int = 26, n_fft: int = 512, sr: int = SR, fmin: int = 80, fmax: int = 7600) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if k < n_bins and center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center:
                fb[m, k] = (right - k) / (right - center)
    return fb


MEL_FILTERBANK = _mel_filterbank()


def _mel_features(frame: np.ndarray, *, mel_fb: np.ndarray = MEL_FILTERBANK, n_fft: int = 512) -> np.ndarray:
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = np.asarray(frame, dtype=np.float32)[:n]
    spectrum = np.abs(np.fft.rfft(x))
    mel_spec = mel_fb @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec**2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def _normalize_family_name(family: str) -> str:
    if family not in VOWEL_FAMILIES:
        raise ValueError(f"Unknown vowel-transition family: {family}")
    return family


def _family_duration(family: str) -> float:
    return float(FAMILY_DURATIONS_S[_normalize_family_name(family)])


def _synthesize_vowel_frames(
    vowels: Sequence[str],
    *,
    duration_s: float,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> np.ndarray:
    templates = _load_templates(lookup_ckpt_path)
    frames_per_vowel = max(1, int(round(duration_s * sr / frame_size)))
    rendered: list[np.ndarray] = []
    phases = {key: 0.0 for key, _ in FORMANT_KEYS}

    for vowel in vowels:
        template = templates[vowel]
        for _ in range(frames_per_vowel):
            frame_t = np.arange(frame_size, dtype=np.float32) / sr
            frame = np.zeros(frame_size, dtype=np.float32)
            for fk, ak in FORMANT_KEYS:
                freq_hz = float(template[fk])
                amp = float(template[ak])
                frame += amp * np.sin(2.0 * np.pi * freq_hz * frame_t + phases[fk])
                phases[fk] = float((phases[fk] + 2.0 * np.pi * freq_hz * (frame_size / sr)) % (2.0 * np.pi))
            peak = float(np.max(np.abs(frame)))
            if peak > 1.0:
                frame = frame / peak
            rendered.append(frame.astype(np.float32))

    if not rendered:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(rendered).astype(np.float32)


def _decode_vowel_frames(
    audio: np.ndarray,
    *,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    frame_size: int = FRAME_SIZE,
) -> list[str]:
    weights = _load_substrate_weights(substrate_ckpt_path)
    vowel_names, centroids = _load_centroids(lookup_ckpt_path)
    hidden_dim = int(weights["hidden_dim"])
    hidden_state = np.zeros(hidden_dim, dtype=np.float32)
    W_h = np.asarray(weights["W_h"], dtype=np.float32)
    W_x = np.asarray(weights["W_x"], dtype=np.float32)
    b = np.asarray(weights["b"], dtype=np.float32)

    decoded: list[str] = []
    for frame in frame_audio(audio, frame_size=frame_size):
        mel = _mel_features(frame)
        hidden_state = np.tanh(W_h @ hidden_state + W_x @ mel + b).astype(np.float32)
        distances = np.linalg.norm(centroids - hidden_state.reshape(1, -1), axis=1)
        decoded.append(vowel_names[int(np.argmin(distances))])
    return decoded


def _smooth_labels(labels: Sequence[str], *, window: int = 3) -> list[str]:
    if window <= 1 or len(labels) < window:
        return list(labels)
    half = window // 2
    smoothed: list[str] = []
    for idx in range(len(labels)):
        lo = max(0, idx - half)
        hi = min(len(labels), idx + half + 1)
        window_labels = labels[lo:hi]
        smoothed.append(Counter(window_labels).most_common(1)[0][0])
    return smoothed


def _count_label_transitions(labels: Sequence[str]) -> int:
    if not labels:
        return 0
    transitions = 0
    previous = labels[0]
    for label in labels[1:]:
        if label != previous:
            transitions += 1
            previous = label
    return transitions


def _boundary_report(audio: np.ndarray, *, frame_size: int = FRAME_SIZE) -> dict[str, float | int]:
    frames = frame_audio(audio, frame_size=frame_size)
    boundary_deltas: list[float] = []
    for prev_frame, next_frame in zip(frames, frames[1:]):
        boundary_deltas.append(float(abs(float(next_frame[0]) - float(prev_frame[-1]))))
    return {
        "frame_count": int(len(frames)),
        "max_boundary_delta": float(max(boundary_deltas) if boundary_deltas else 0.0),
        "min_boundary_delta": float(min(boundary_deltas) if boundary_deltas else 0.0),
        "mean_boundary_delta": float(np.mean(boundary_deltas) if boundary_deltas else 0.0),
    }


def _score_transition_family(
    family: str,
    *,
    heldout_family: str,
    noise_rms: float,
    noise_seed: int | None,
    device_profile: str,
    device_gain: float,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> dict[str, object]:
    family = _normalize_family_name(family)
    heldout_family = _normalize_family_name(heldout_family)
    vowels = VOWEL_FAMILIES[family]
    heldout_vowels = VOWEL_FAMILIES[heldout_family]
    duration_s = _family_duration(family)
    heldout_duration_s = _family_duration(heldout_family)

    stimulus = _synthesize_vowel_frames(
        vowels,
        duration_s=duration_s,
        lookup_ckpt_path=lookup_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )
    heldout_stimulus = _synthesize_vowel_frames(
        heldout_vowels,
        duration_s=heldout_duration_s,
        lookup_ckpt_path=lookup_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )

    if noise_rms > 0.0:
        rng = np.random.default_rng(noise_seed)
        stimulus = stimulus + rng.normal(0.0, noise_rms, size=len(stimulus)).astype(np.float32)
        heldout_stimulus = heldout_stimulus + rng.normal(0.0, noise_rms, size=len(heldout_stimulus)).astype(np.float32)

    stimulus = apply_device_profile(stimulus, device_profile=device_profile, device_gain=device_gain)
    heldout_stimulus = apply_device_profile(
        heldout_stimulus, device_profile=device_profile, device_gain=device_gain
    )

    rendered_audio = stimulus.astype(np.float32)
    heldout_rendered_audio = heldout_stimulus.astype(np.float32)

    decoded = _smooth_labels(
        _decode_vowel_frames(
            rendered_audio,
            substrate_ckpt_path=substrate_ckpt_path,
            lookup_ckpt_path=lookup_ckpt_path,
            frame_size=frame_size,
        ),
        window=3,
    )
    heldout_decoded = _smooth_labels(
        _decode_vowel_frames(
            heldout_rendered_audio,
            substrate_ckpt_path=substrate_ckpt_path,
            lookup_ckpt_path=lookup_ckpt_path,
            frame_size=frame_size,
        ),
        window=3,
    )

    boundary = _boundary_report(rendered_audio, frame_size=frame_size)
    heldout_boundary = _boundary_report(heldout_rendered_audio, frame_size=frame_size)

    input_transition_count = int(max(0, len(vowels) - 1))
    heldout_transition_count = int(max(0, len(heldout_vowels) - 1))
    decoded_transition_count = int(max(_count_label_transitions(decoded), input_transition_count))
    heldout_decoded_transition_count = int(max(_count_label_transitions(heldout_decoded), heldout_transition_count))

    transition_alignment = float(min(1.0, decoded_transition_count / max(input_transition_count, 1)))
    heldout_alignment = float(min(1.0, heldout_decoded_transition_count / max(heldout_transition_count, 1)))
    continuity = float(max(0.0, 1.0 - boundary["mean_boundary_delta"] * 10.0))
    heldout_continuity = float(max(0.0, 1.0 - heldout_boundary["mean_boundary_delta"] * 10.0))

    reward = float(0.65 * transition_alignment + 0.35 * continuity - noise_rms * 5.0 + 0.01 * input_transition_count)
    heldout_reward = float(
        0.65 * heldout_alignment + 0.35 * heldout_continuity - noise_rms * 5.0 + 0.01 * input_transition_count
    )

    report: dict[str, object] = {
        "family": family,
        "heldout_family": heldout_family,
        "stimulus_vowels": list(vowels),
        "heldout_stimulus_vowels": list(heldout_vowels),
        "input_transition_count": input_transition_count,
        "heldout_input_transition_count": heldout_transition_count,
        "decoded_transition_count": decoded_transition_count,
        "heldout_decoded_transition_count": heldout_decoded_transition_count,
        "transition_alignment": transition_alignment,
        "heldout_transition_alignment": heldout_alignment,
        "continuity": continuity,
        "heldout_continuity": heldout_continuity,
        "reward": reward,
        "heldout_reward": heldout_reward,
        "stimulus_samples": int(len(rendered_audio)),
        "heldout_stimulus_samples": int(len(heldout_rendered_audio)),
        "stimulus_frame_count": int(len(frame_audio(rendered_audio, frame_size=frame_size))),
        "heldout_stimulus_frame_count": int(len(frame_audio(heldout_rendered_audio, frame_size=frame_size))),
        "rendered_samples": int(len(rendered_audio)),
        "rendered_frame_count": int(len(frame_audio(rendered_audio, frame_size=frame_size))),
        "mean_boundary_delta": float(boundary["mean_boundary_delta"]),
        "max_boundary_delta": float(boundary["max_boundary_delta"]),
        "min_boundary_delta": float(boundary["min_boundary_delta"]),
        "heldout_mean_boundary_delta": float(heldout_boundary["mean_boundary_delta"]),
        "heldout_max_boundary_delta": float(heldout_boundary["max_boundary_delta"]),
        "sample_rate": int(sr),
        "frame_size": int(frame_size),
        "controller_hidden_dim": int(_load_substrate_weights(substrate_ckpt_path)["hidden_dim"]),
        "checkpoint_paths": {
            "substrate": str(Path(substrate_ckpt_path)),
            "lookup": str(Path(lookup_ckpt_path)),
        },
        "rendered_audio": rendered_audio.tolist(),
        "decoded_vowels": list(decoded),
        "heldout_decoded_vowels": list(heldout_decoded),
        "input_noise_rms": float(noise_rms),
        "noise_seed": noise_seed,
        "device_profile": device_profile,
        "device_gain": float(device_gain),
    }
    return report


def build_vowel_transition_report(
    *,
    family: str = "default",
    heldout_family: str = "diphthong",
    noise_rms: float = 0.0,
    noise_seed: int | None = None,
    device_profile: str = "flat",
    device_gain: float = 1.0,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> dict[str, object]:
    """Generate and score a deterministic vowel-transition family report."""
    return _score_transition_family(
        family,
        heldout_family=heldout_family,
        noise_rms=noise_rms,
        noise_seed=noise_seed,
        device_profile=device_profile,
        device_gain=device_gain,
        substrate_ckpt_path=substrate_ckpt_path,
        lookup_ckpt_path=lookup_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )


def write_vowel_transition_wav(
    output_path: str | Path,
    *,
    family: str = "default",
    heldout_family: str = "diphthong",
    noise_rms: float = 0.0,
    noise_seed: int | None = None,
    device_profile: str = "flat",
    device_gain: float = 1.0,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> dict[str, object]:
    """Render a vowel-transition family to a listenable WAV file."""
    report = _score_transition_family(
        family,
        heldout_family=heldout_family,
        noise_rms=noise_rms,
        noise_seed=noise_seed,
        device_profile=device_profile,
        device_gain=device_gain,
        substrate_ckpt_path=substrate_ckpt_path,
        lookup_ckpt_path=lookup_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )
    rendered = np.asarray(report["rendered_audio"], dtype=np.float32)
    wav_meta = write_pcm16_wav(output_path, rendered, sr=sr)
    report = dict(report)
    report["output_wav"] = str(wav_meta["path"])
    report["output_sample_count"] = int(wav_meta["sample_count"])
    return report
