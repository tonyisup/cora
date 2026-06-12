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
from typing import Any, Sequence, cast

import numpy as np

from m9_sound.temporal import FRAME_SIZE, SR, apply_device_profile, frame_audio, write_pcm16_wav

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSTRATE_CKPT = REPO_ROOT / "data" / "m9_substrate_spectral_v11.ckpt.json"
DEFAULT_LOOKUP_CKPT = REPO_ROOT / "data" / "m9_formant_lookup_v15.json"

VOWEL_FAMILIES: dict[str, list[str]] = {
    "default": ["i", "E", "u"],
    "diphthong": ["i", "E", "a", "O", "u"],
    "phase_triad": ["i", "E", "a", "i"],
    "phase_arc": ["i", "O", "i"],
    "wide_sweep": ["i", "E", "a", "O", "u", "O", "a"],
    "long_arc": ["i", "E", "a", "O", "u", "O", "a", "E", "i"],
    "formant_rich": ["i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "a", "O", "i"],
    "dense_cycle": ["i", "E", "a", "O", "u", "O", "a", "E", "i", "E", "a", "O", "u", "O", "a", "E", "i"],
    "ultra_dense": ["i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "a", "O", "E", "i", "O", "u", "a", "E", "i", "O", "i", "i"],
    "hyper_dense": ["i", "E", "a", "O", "u", "i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "O", "a", "E", "i", "O", "u", "E", "a", "i", "O", "u", "i", "i"],
    "mega_dense": ["i", "E", "a", "O", "u", "i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "O", "a", "E", "i", "O", "u", "E", "a", "i", "O", "u", "i", "E", "a", "O", "u", "E", "a", "i"],
    "max_dense": ["i", "E", "a", "O", "u", "i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "O", "a", "E", "i", "O", "u", "E", "a", "i", "O", "u", "i", "E", "a", "O", "u", "E", "a", "i", "E", "a", "O", "u", "i"],
    "ultra_max_dense": ["i", "E", "a", "O", "u", "i", "E", "a", "O", "u", "O", "a", "E", "i", "u", "O", "a", "E", "i", "O", "u", "E", "a", "i", "O", "u", "i", "E", "a", "O", "u", "E", "a", "i", "E", "a", "O", "u", "i", "O", "u", "E", "a", "i"],
}

FAMILY_DURATIONS_S: dict[str, float] = {
    "default": 0.60,
    "diphthong": 0.40,
    "phase_triad": 0.90,
    "phase_arc": 0.60,
    "wide_sweep": 0.60,
    "long_arc": 0.50,
    "formant_rich": 0.50,
    "dense_cycle": 0.50,
    "ultra_dense": 0.50,
    "hyper_dense": 0.50,
    "mega_dense": 0.50,
    "max_dense": 0.50,
    "ultra_max_dense": 0.50,
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


MEL_FILTERBANK = _mel_filterbank(n_mels=32)


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


def _family_duration(family: str, *, duration_scale: float = 1.0) -> float:
    return float(FAMILY_DURATIONS_S[_normalize_family_name(family)] * duration_scale)


def _synthesize_vowel_frames(
    vowels: Sequence[str],
    *,
    duration_s: float,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    templates = _load_templates(lookup_ckpt_path)
    n_vowels = len(vowels)
    # Each vowel gets frames_per_vowel frames of stable body
    # Plus a crossfade to the next vowel
    n_stable = max(2, int(round(duration_s * sr / frame_size / n_vowels * 0.7)))
    n_crossfade = min(n_stable, max(7, int(round(duration_s * sr / frame_size / n_vowels * 0.3))))
    rendered: list[np.ndarray] = []
    segment_meta: list[dict[str, object]] = []
    phases = {key: 0.0 for key, _ in FORMANT_KEYS}

    if not vowels:
        return np.zeros(0, dtype=np.float32), segment_meta

    # Render each vowel body, then crossfade to next
    for vowel_idx, vowel in enumerate(vowels):
        template = templates[vowel]
        # Render stable vowel body
        for frame_index in range(n_stable):
            frame_t = np.arange(frame_size, dtype=np.float32) / sr
            frame = np.zeros(frame_size, dtype=np.float32)
            for fk, ak in FORMANT_KEYS:
                frame += template[ak] * np.sin(2.0 * np.pi * template[fk] * frame_t + phases[fk])
                phases[fk] = float((phases[fk] + 2.0 * np.pi * template[fk] * (frame_size / sr)) % (2.0 * np.pi))
            rendered.append(frame.astype(np.float32))
        segment_meta.append({"vowel": vowel, "frame_count": n_stable})

        # Crossfade to next vowel (if not last)
        if vowel_idx < n_vowels - 1:
            next_template = templates[vowels[vowel_idx + 1]]
            for frame_index in range(n_crossfade):
                # Smoothraised-cosine crossfade
                t_raw = frame_index / max(1, n_crossfade - 1) if n_crossfade > 1 else 1.0
                # Raised cosine for smoother transition
                t = 0.5 * (1 - np.cos(np.pi * t_raw))
                # Amplitude envelope: fade out current, fade in next
                amp_out = 1.0 - t
                amp_in = t
                frame_t = np.arange(frame_size, dtype=np.float32) / sr
                frame = np.zeros(frame_size, dtype=np.float32)
                # Interpolate formant frequencies and amplitudes
                for fk, ak in FORMANT_KEYS:
                    freq_hz = float((1.0 - t) * template[fk] + t * next_template[fk])
                    amp = amp_out * float(template[ak]) + amp_in * float(next_template[ak])
                    frame += amp * np.sin(2.0 * np.pi * freq_hz * frame_t + phases[fk])
                    phases[fk] = float((phases[fk] + 2.0 * np.pi * freq_hz * (frame_size / sr)) % (2.0 * np.pi))
                rendered.append(frame.astype(np.float32))
            segment_meta.append({"vowel": f"{vowel}->{vowels[vowel_idx+1]}", "frame_count": n_crossfade})

    if not rendered:
        return np.zeros(0, dtype=np.float32), segment_meta

    # Normalize the entire rendered signal once, not per-frame
    peak = float(np.max(np.abs(np.concatenate(rendered))))
    if peak > 1.0:
        rendered = [frame / peak for frame in rendered]

    return np.concatenate(rendered).astype(np.float32), segment_meta


def _decode_vowel_frames(
    audio: np.ndarray,
    *,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    frame_size: int = FRAME_SIZE,
    silence_threshold: float = 0.0,
    consonant_gate: float = 0.0,
    center_mel: bool = False,
) -> list[str]:
    weights = _load_substrate_weights(substrate_ckpt_path)
    vowel_names, centroids = _load_centroids(lookup_ckpt_path)
    hidden_dim = int(weights["hidden_dim"])
    hidden_state = np.zeros(hidden_dim, dtype=np.float32)
    W_h = np.asarray(weights["W_h"], dtype=np.float32)
    W_x = np.asarray(weights["W_x"], dtype=np.float32)
    b = np.asarray(weights["b"], dtype=np.float32)

    # First pass: compute all mel features if centering
    if center_mel:
        all_mel = []
        for frame in frame_audio(audio, frame_size=frame_size):
            all_mel.append(_mel_features(frame))
        if all_mel:
            mean_mel = np.mean(all_mel, axis=0, keepdims=True)

    decoded: list[str] = []
    frame_idx = 0
    for frame in frame_audio(audio, frame_size=frame_size):
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if rms < silence_threshold:
            decoded.append("_")
        else:
            mel = _mel_features(frame)
            if center_mel and all_mel:
                mel = mel - mean_mel[0]
            hidden_state = np.tanh(W_h @ hidden_state + W_x @ mel + b).astype(np.float32)
            distances = np.linalg.norm(centroids - hidden_state.reshape(1, -1), axis=1)
            best_idx = int(np.argmin(distances))
            confidence = float(distances[best_idx])
            if consonant_gate > 0.0 and confidence > consonant_gate:
                decoded.append("_")
            else:
                decoded.append(vowel_names[best_idx])
        frame_idx += 1
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
    duration_scale: float = 1.0,
) -> dict[str, object]:
    family = _normalize_family_name(family)
    heldout_family = _normalize_family_name(heldout_family)
    vowels = VOWEL_FAMILIES[family]
    heldout_vowels = VOWEL_FAMILIES[heldout_family]
    duration_s = _family_duration(family, duration_scale=duration_scale)
    heldout_duration_s = _family_duration(heldout_family, duration_scale=duration_scale)

    stimulus, transition_segments = _synthesize_vowel_frames(
        vowels,
        duration_s=duration_s,
        lookup_ckpt_path=lookup_ckpt_path,
        sr=sr,
        frame_size=frame_size,
    )
    heldout_stimulus, heldout_transition_segments = _synthesize_vowel_frames(
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

    segment_frame_count = cast(int, cast(dict[str, object], transition_segments[0])["frame_count"])
    report: dict[str, object] = {
        "family": family,
        "heldout_family": heldout_family,
        "stimulus_vowels": list(vowels),
        "heldout_stimulus_vowels": list(heldout_vowels),
        "stimulus_unique_vowel_count": int(len(set(vowels))),
        "heldout_unique_vowel_count": int(len(set(heldout_vowels))),
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
        "transition_segments": list(transition_segments),
        "transition_segment_count": int(len(transition_segments)),
        "segment_frame_count": segment_frame_count,
        "heldout_transition_segments": list(heldout_transition_segments),
        "heldout_transition_segment_count": int(len(heldout_transition_segments)),
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
    duration_scale: float = 1.0,
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
        duration_scale=duration_scale,
    )


def decode_vowel_transition_audio(
    audio: np.ndarray,
    *,
    substrate_ckpt_path: str | Path = DEFAULT_SUBSTRATE_CKPT,
    lookup_ckpt_path: str | Path = DEFAULT_LOOKUP_CKPT,
    frame_size: int = FRAME_SIZE,
    smooth_window: int = 3,
    silence_threshold: float = 0.0,
    consonant_gate: float = 0.0,
    center_mel: bool = False,
) -> dict[str, object]:
    """Decode a rendered vowel-transition audio buffer into a compact report."""
    decoded = _smooth_labels(
        _decode_vowel_frames(
            audio,
            substrate_ckpt_path=substrate_ckpt_path,
            lookup_ckpt_path=lookup_ckpt_path,
            frame_size=frame_size,
            silence_threshold=silence_threshold,
            consonant_gate=consonant_gate,
            center_mel=center_mel,
        ),
        window=smooth_window,
    )
    boundary = _boundary_report(np.asarray(audio, dtype=np.float32), frame_size=frame_size)
    decoded_transition_count = int(_count_label_transitions(decoded))

    # Extract silence-separated vowel segments
    vowel_segments: list[dict[str, object]] = []
    current_vowel = ""
    current_start = 0
    for idx, label in enumerate(decoded):
        if label == "_":
            if current_vowel:
                vowel_segments.append({
                    "vowel": current_vowel,
                    "start_frame": current_start,
                    "end_frame": idx,
                    "duration_frames": idx - current_start,
                })
                current_vowel = ""
        else:
            if not current_vowel:
                current_vowel = label
                current_start = idx
            elif label != current_vowel:
                vowel_segments.append({
                    "vowel": current_vowel,
                    "start_frame": current_start,
                    "end_frame": idx,
                    "duration_frames": idx - current_start,
                })
                current_vowel = label
                current_start = idx
    if current_vowel:
        vowel_segments.append({
            "vowel": current_vowel,
            "start_frame": current_start,
            "end_frame": len(decoded),
            "duration_frames": len(decoded) - current_start,
        })

    return {
        "decoded_vowels": list(decoded),
        "decoded_transition_count": decoded_transition_count,
        "decoded_unique_vowel_count": int(len(set(v for v in decoded if v != "_"))) if decoded else 0,
        "frame_count": int(len(frame_audio(audio, frame_size=frame_size))),
        "mean_boundary_delta": float(boundary["mean_boundary_delta"]),
        "max_boundary_delta": float(boundary["max_boundary_delta"]),
        "min_boundary_delta": float(boundary["min_boundary_delta"]),
        "continuity": float(max(0.0, 1.0 - boundary["mean_boundary_delta"] * 10.0)),
        "vowel_segments": vowel_segments,
        "silence_threshold": silence_threshold,
    }


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
    duration_scale: float = 1.0,
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
        duration_scale=duration_scale,
    )
    rendered = np.asarray(report["rendered_audio"], dtype=np.float32)
    wav_meta = write_pcm16_wav(output_path, rendered, sr=sr)
    report = dict(report)
    report["output_wav"] = str(wav_meta["path"])
    report["output_sample_count"] = int(wav_meta["sample_count"])
    return report
