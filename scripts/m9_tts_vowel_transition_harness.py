#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""M9 TTS vowel-transition speaker harness.

This harness synthesizes short vowel-transition phrases with macOS TTS, plays
those phrases through a selected output device, captures them from the mic, and
scores the recorded audio with the vowel-transition decoder.

It is meant for external-speaker stress testing, not as a promotion baseline.
Use it to compare output devices, playback levels, and phrase sets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import VOWEL_FAMILIES, build_vowel_transition_report, decode_vowel_transition_audio, write_pcm16_wav

SR = 16_000
DEFAULT_INPUT_DEVICE = 1
DEFAULT_OUTPUT_DEVICE = 2
DEFAULT_VOICE = "Samantha"
DEFAULT_RATE = "-20%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_LEAD_IN_S = 0.25
DEFAULT_TAIL_S = 0.25
DEFAULT_FRAME_SIZE = 480
DEFAULT_MIN_TRANSITIONS = 2
DEFAULT_MIN_CONTINUITY = 0.88
DEFAULT_MONITOR_OUTPUT_DEVICE = 3
DEFAULT_MONITOR_MODE = "project"
DEFAULT_MONITOR_FAMILY = "auto"

DEFAULT_PHRASES = [
    "beat bet hot caught boot",
    "boot caught hot bet beat",
    "beat hot bet boot caught",
    "caught boot bet hot beat",
    "bet beat boot caught hot",
    "hot caught beat boot bet",
]


def _read_wav_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels in {path}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV, got sample width {sampwidth} in {path}")
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1)


def _write_tts_audio_to_wav(
    *,
    text: str,
    output_wav: Path,
    voice: str,
    rate: str,
    pitch: str,
) -> np.ndarray:
    """Use macOS `say` to synthesize text, then convert it to 16 kHz WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m9-tts-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        aiff_path = tmpdir_path / "tts.aiff"
        subprocess.run(
            ["say", "-v", voice, "-r", rate.replace("%", "") if rate.endswith("%") else rate, "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_path), str(output_wav)],
            check=True,
        )
    return _read_wav_pcm(output_wav)


def _capture_playback_with_playrec(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - only on real hardware
        raise RuntimeError("sounddevice is required for live capture") from exc

    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    pad_lead = np.zeros(int(round(lead_in_s * SR)), dtype=np.float32)
    pad_tail = np.zeros(int(round(tail_s * SR)), dtype=np.float32)
    playback = np.concatenate([pad_lead, pcm, pad_tail])

    try:
        captured = sd.playrec(
            playback,
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=(input_device, output_device),
            blocking=True,
        )
    except TypeError:
        sd.default.device = (input_device, output_device)
        captured = sd.playrec(
            playback,
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocking=True,
        )
    captured = np.asarray(captured, dtype=np.float32).reshape(-1)
    return captured, {
        "playback_samples": int(len(playback)),
        "capture_samples": int(len(captured)),
        "lead_in_s": float(lead_in_s),
        "tail_s": float(tail_s),
        "input_device": int(input_device),
        "output_device": int(output_device),
    }


def _default_capture_audio_fn(
    pcm: np.ndarray,
    *,
    input_device: int,
    output_device: int,
    lead_in_s: float,
    tail_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _capture_playback_with_playrec(
        pcm,
        input_device=input_device,
        output_device=output_device,
        lead_in_s=lead_in_s,
        tail_s=tail_s,
    )


def _play_pcm_on_device(pcm: np.ndarray, *, output_device: int, label: str = "monitor") -> dict[str, Any]:
    """Play a mono PCM buffer on an output device and wait for completion."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - only on real hardware
        raise RuntimeError("sounddevice is required for monitor playback") from exc

    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    sd.play(pcm, samplerate=SR, device=output_device, blocking=True)
    return {
        "output_device": int(output_device),
        "label": label,
        "sample_count": int(len(pcm)),
    }


def _default_monitor_audio_fn(pcm: np.ndarray, *, output_device: int) -> dict[str, Any]:
    return _play_pcm_on_device(pcm, output_device=output_device, label="captured")


PROJECT_MONITOR_FAMILY_ORDER = [
    "phase_arc",
    "phase_triad",
    "diphthong",
    "wide_sweep",
    "long_arc",
    "formant_rich",
    "dense_cycle",
    "ultra_dense",
    "hyper_dense",
    "mega_dense",
    "max_dense",
    "ultra_max_dense",
]


def _family_rank(family: str) -> int:
    try:
        return PROJECT_MONITOR_FAMILY_ORDER.index(family)
    except ValueError:
        return -1


def _project_family_from_phrase_token_count(token_count: int) -> str:
    count = max(1, int(token_count))
    if count <= 1:
        return "phase_arc"
    if count == 2:
        return "phase_triad"
    if count == 3:
        return "diphthong"
    if count == 4:
        return "wide_sweep"
    if count == 5:
        return "long_arc"
    if count == 6:
        return "formant_rich"
    if count <= 8:
        return "dense_cycle"
    if count <= 10:
        return "ultra_dense"
    if count <= 12:
        return "hyper_dense"
    if count <= 16:
        return "mega_dense"
    if count <= 20:
        return "max_dense"
    return "ultra_max_dense"


def _project_family_from_decoded_shape(decoded_unique_vowel_count: int, decoded_transition_count: int) -> str:
    unique_count = max(1, int(decoded_unique_vowel_count))
    transition_count = max(0, int(decoded_transition_count))

    if unique_count >= 8:
        return "long_arc"
    if unique_count >= 5:
        return "diphthong"
    if transition_count >= 7:
        return "phase_triad"
    return "phase_arc"


def _select_project_monitor_family(phrase_token_count: int, decoded_unique_vowel_count: int, decoded_transition_count: int) -> tuple[str, str, str]:
    phrase_family = _project_family_from_phrase_token_count(phrase_token_count)
    decoded_family = _project_family_from_decoded_shape(decoded_unique_vowel_count, decoded_transition_count)
    resolved_family = phrase_family if _family_rank(phrase_family) >= _family_rank(decoded_family) else decoded_family
    return resolved_family, phrase_family, decoded_family


def _generate_project_imitation_audio(
    *,
    family: str,
    heldout_family: str,
    device_profile: str,
    device_gain: float,
    monitor_family: str,
    noise_rms: float,
    noise_seed: int | None,
    duration_scale: float,
    frame_size: int,
    output_wav: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render the project's own imitation audio for listening."""
    report = build_vowel_transition_report(
        family=family,
        heldout_family=heldout_family,
        noise_rms=noise_rms,
        noise_seed=noise_seed,
        device_profile=device_profile,
        device_gain=device_gain,
        frame_size=frame_size,
        duration_scale=duration_scale,
    )
    rendered_pcm = np.asarray(report["rendered_audio"], dtype=np.float32).reshape(-1)
    rendered_meta = write_pcm16_wav(output_wav, rendered_pcm, sr=SR)
    return rendered_pcm, {
        "family": family,
        "monitor_family": monitor_family,
        "heldout_family": heldout_family,
        "device_profile": device_profile,
        "device_gain": float(device_gain),
        "duration_scale": float(duration_scale),
        "sample_count": int(rendered_meta["sample_count"]),
        "wav_path": str(rendered_meta["path"]),
        "decoded_transition_count": int(report["input_transition_count"]),
    }


def _generate_decoded_imitation_audio(
    *,
    decoded_vowels: list[str],
    device_profile: str,
    device_gain: float,
    noise_rms: float,
    noise_seed: int | None,
    frame_size: int,
    output_wav: Path,
    vowel_timing: list[tuple[str, float]] | None = None,
    silence_timing: list[float] | None = None,
    tts_duration: float = 2.0,
    hold_duration: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render imitation audio that follows the decoded TTS vowel sequence.

    If vowel_timing is provided, each vowel body is rendered for its
    matched duration (with a minimum of 0.1s). Otherwise falls back to
    uniform timing based on tts_duration.

    If silence_timing is provided, silence gaps are inserted between
    vowels to match the TTS rhythm.

    If hold_duration > 0, each vowel is held for exactly that duration
    (useful for "booooot" style sustained vowels that give the substrate
    more time to classify).
    """
    from m9_sound.vowel_transitions import write_pcm16_wav

    # Build a vowel sequence from the decoded vowels
    unique_vowels = list(dict.fromkeys(decoded_vowels))
    if not unique_vowels:
        unique_vowels = ["i", "E", "u"]

    # Use per-vowel timing from TTS if available, otherwise uniform
    if vowel_timing:
        vowel_sequence = [v for v, _ in vowel_timing]
        per_vowel_duration = [max(0.1, d) for _, d in vowel_timing]
    else:
        vowels_per_second = 3.0
        n_vowels = max(3, int(round(tts_duration * vowels_per_second)))
        vowel_sequence = [unique_vowels[i % len(unique_vowels)] for i in range(n_vowels)]
        per_vowel_duration = [tts_duration / n_vowels] * n_vowels

    fs = SR
    fs_frame = frame_size

    # Synthesize using the vowel sequence directly
    from m9_sound.vowel_transitions import DEFAULT_LOOKUP_CKPT as _DEFAULT_LOOKUP_CKPT
    templates = _load_templates(_DEFAULT_LOOKUP_CKPT)

    rendered = []
    phases = {fk: 0.0 for fk, _ in [("f1","a1"),("f2","a2"),("f3","a3")]}

    for idx, vowel in enumerate(vowel_sequence):
        if vowel not in templates:
            vowel = "i"
        t = templates[vowel]
        if hold_duration > 0:
            vowel_dur = hold_duration
        else:
            vowel_dur = per_vowel_duration[idx] if idx < len(per_vowel_duration) else 0.3
        n_stable = max(2, int(round(vowel_dur * fs / fs_frame * 0.7)))
        n_crossfade = min(n_stable, max(4, int(round(vowel_dur * fs / fs_frame * 0.3))))
        for fi in range(n_stable):
            frame_t = np.arange(fs_frame, dtype=np.float32) / fs
            frame = np.zeros(fs_frame, dtype=np.float32)
            for fk, ak in [("f1","a1"),("f2","a2"),("f3","a3")]:
                frame += t[ak] * np.sin(2*np.pi*t[fk]*frame_t + phases[fk])
                phases[fk] = (phases[fk] + 2*np.pi*t[fk]*(fs_frame/fs)) % (2*np.pi)
            rendered.append(frame.astype(np.float32))

        # Insert silence gap between vowels to match TTS rhythm
        if silence_timing and idx < len(silence_timing):
            gap_s = silence_timing[idx]
            gap_samples = int(round(gap_s * fs))
            if gap_samples > 0:
                rendered.append(np.zeros(gap_samples, dtype=np.float32))

        if idx < len(vowel_sequence) - 1:
            next_v = vowel_sequence[idx + 1]
            if next_v not in templates:
                next_v = "i"
            nt = templates[next_v]
            for fi in range(n_crossfade):
                t_raw = fi / max(1, n_crossfade - 1) if n_crossfade > 1 else 1.0
                alpha = 0.5 * (1 - np.cos(np.pi * t_raw))
                ao = 1.0 - alpha
                ai = alpha
                frame_t = np.arange(fs_frame, dtype=np.float32) / fs
                frame = np.zeros(fs_frame, dtype=np.float32)
                for fk, ak in [("f1","a1"),("f2","a2"),("f3","a3")]:
                    freq = (1-alpha)*t[fk] + alpha*nt[fk]
                    amp = ao*t[ak] + ai*nt[ak]
                    frame += amp * np.sin(2*np.pi*freq*frame_t + phases[fk])
                    phases[fk] = (phases[fk] + 2*np.pi*freq*(fs_frame/fs)) % (2*np.pi)
                rendered.append(frame.astype(np.float32))

    if rendered:
        peak = float(np.max(np.abs(np.concatenate(rendered))))
        if peak > 1.0:
            rendered = [f/peak for f in rendered]

    rendered_pcm = np.concatenate(rendered).astype(np.float32)
    rendered_pcm = np.asarray(rendered_pcm, dtype=np.float32).reshape(-1)

    if noise_rms > 0.0:
        rng = np.random.default_rng(noise_seed)
        rendered_pcm = rendered_pcm + rng.normal(0.0, noise_rms, size=len(rendered_pcm)).astype(np.float32)

    from m9_sound.temporal import apply_device_profile
    rendered_pcm = apply_device_profile(rendered_pcm, device_profile=device_profile, device_gain=device_gain)

    rendered_meta = write_pcm16_wav(output_wav, rendered_pcm, sr=fs)
    return rendered_pcm, {
        "family": "decoded",
        "vowel_sequence": vowel_sequence,
        "device_profile": device_profile,
        "device_gain": float(device_gain),
        "sample_count": int(rendered_meta["sample_count"]),
        "wav_path": str(rendered_meta["path"]),
    }


def _extract_vowel_timing(decoded_vowels, *, frame_size=480, sr=16000):
    """Extract per-vowel durations from decoded vowel sequence.

    Returns list of (vowel, duration_s) tuples, where each duration
    is the number of frames that vowel was active.
    """
    if not decoded_vowels:
        return [("i", 0.3), ("E", 0.3), ("u", 0.3)]

    timing = []
    current = decoded_vowels[0]
    count = 1
    for v in decoded_vowels[1:]:
        if v == current:
            count += 1
        else:
            timing.append((current, count * frame_size / sr))
            current = v
            count = 1
    timing.append((current, count * frame_size / sr))
    return timing


def _extract_silence_timing(decoded: dict[str, object], *, frame_size=480, sr=16000) -> list[float]:
    """Extract silence gap durations between vowel segments."""
    segments = decoded.get("vowel_segments", [])
    if len(segments) < 2:
        return []
    timing = []
    for idx in range(len(segments) - 1):
        end_frame = segments[idx]["end_frame"]
        start_frame = segments[idx + 1]["start_frame"]
        gap_frames = start_frame - end_frame
        if gap_frames > 0:
            timing.append(gap_frames * frame_size / sr)
        else:
            timing.append(0.0)
    return timing


def _estimate_silence_rms(audio: np.ndarray, *, frame_size=480) -> float:
    """Estimate the RMS level of silence in the audio.

    Uses the 10th percentile of frame RMS values as the silence floor.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return 0.001
    rms_values = []
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)
    rms_arr = np.array(rms_values)
    return float(np.percentile(rms_arr, 10))


def _load_templates(lookup_ckpt_path):
    import json
    from pathlib import Path
    data = json.loads(Path(lookup_ckpt_path).read_text())
    return {v: dict(data["lookup"][v]["template"]) for v in data["vowel_names"]}


def _parse_phrases(values: Sequence[str]) -> list[str]:
    phrases = [value.strip() for value in values if value.strip()]
    if not phrases:
        raise ValueError("At least one phrase is required")
    return phrases


def run_tts_vowel_transition_harness(
    *,
    output_dir: str | Path,
    output: str | Path | None = None,
    phrases: Sequence[str] | None = None,
    repeats: int = 1,
    input_device: int = DEFAULT_INPUT_DEVICE,
    output_device: int = DEFAULT_OUTPUT_DEVICE,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
    lead_in_s: float = DEFAULT_LEAD_IN_S,
    tail_s: float = DEFAULT_TAIL_S,
    threshold_transitions: int = DEFAULT_MIN_TRANSITIONS,
    threshold_continuity: float = DEFAULT_MIN_CONTINUITY,
    frame_size: int = DEFAULT_FRAME_SIZE,
    dry_run: bool = False,
    monitor_output_device: int | None = DEFAULT_MONITOR_OUTPUT_DEVICE,
    monitor_mode: str = DEFAULT_MONITOR_MODE,
    monitor_family: str = DEFAULT_MONITOR_FAMILY,
    hold_duration: float = 0.0,
    tts_audio_fn: Callable[..., np.ndarray] | None = None,
    capture_audio_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]] | None = None,
    monitor_audio_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output) if output is not None else output_dir / "report.json"
    phrase_list = _parse_phrases(DEFAULT_PHRASES if phrases is None else phrases)
    repeats = max(1, int(repeats))

    synth_fn = tts_audio_fn or _write_tts_audio_to_wav
    capture_fn = capture_audio_fn or _default_capture_audio_fn
    monitor_fn = monitor_audio_fn or _default_monitor_audio_fn if monitor_mode != "off" else None

    runs: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        for phrase_idx, phrase in enumerate(phrase_list, start=1):
            run_dir = output_dir / f"repeat-{repeat_idx + 1:02d}" / f"phrase-{phrase_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tts_wav = run_dir / "tts.wav"
            captured_wav = run_dir / "captured.wav"

            target_pcm = synth_fn(
                text=phrase,
                output_wav=tts_wav,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            target_pcm = np.asarray(target_pcm, dtype=np.float32).reshape(-1)
            if dry_run:
                captured_pcm = target_pcm.copy()
                capture_meta = {
                    "playback_samples": int(len(target_pcm)),
                    "capture_samples": int(len(target_pcm)),
                    "lead_in_s": float(lead_in_s),
                    "tail_s": float(tail_s),
                    "input_device": int(input_device),
                    "output_device": int(output_device),
                }
            else:
                captured_pcm, capture_meta = capture_fn(
                    pcm=target_pcm,
                    input_device=input_device,
                    output_device=output_device,
                    lead_in_s=lead_in_s,
                    tail_s=tail_s,
                )

            captured_pcm = np.asarray(captured_pcm, dtype=np.float32).reshape(-1)
            captured_meta = write_pcm16_wav(captured_wav, captured_pcm, sr=SR)
            # Apply HP100 preprocessing to match v9 substrate training
            from scipy.signal import butter, sosfilt
            hp100_sos = butter(4, 100, btype='high', fs=SR, output='sos')
            captured_proc = sosfilt(hp100_sos, captured_pcm).astype(np.float32)
            # Per-frame RMS normalization to remove energy as a cue
            # This forces the substrate to use spectral shape for vowel classification
            frame_size = FRAME_SIZE
            n_frames = len(captured_proc) // frame_size
            for i in range(n_frames):
                frame = captured_proc[i*frame_size:(i+1)*frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                if rms > 1e-6:
                    frame[:] = frame / rms
            # Also apply to trailing samples
            remainder = len(captured_proc) % frame_size
            if remainder > 0:
                frame = captured_proc[n_frames*frame_size:]
                rms = np.sqrt(np.mean(frame ** 2))
                if rms > 1e-6:
                    frame[:] = frame / rms
            silence_rms = _estimate_silence_rms(captured_proc, frame_size=frame_size)
            decoded = decode_vowel_transition_audio(
                captured_proc, frame_size=frame_size,
                silence_threshold=silence_rms * 8.0,
                consonant_gate=0.0,
                center_mel=True,
            )
            transition_count = int(decoded["decoded_transition_count"])
            continuity = float(decoded["continuity"])
            phrase_token_count = len([token for token in phrase.split() if token])
            resolved_monitor_family = monitor_family
            phrase_monitor_family = _project_family_from_phrase_token_count(phrase_token_count)
            decoded_monitor_family = _project_family_from_decoded_shape(
                int(decoded["decoded_unique_vowel_count"]),
                transition_count,
            )
            monitor_family_source = "explicit"
            if monitor_mode == "project" and monitor_family == "auto":

                resolved_monitor_family, phrase_monitor_family, decoded_monitor_family = _select_project_monitor_family(
                    phrase_token_count,
                    int(decoded["decoded_unique_vowel_count"]),
                    transition_count,
                )
                monitor_family_source = "phrase_shape" if _family_rank(phrase_monitor_family) >= _family_rank(decoded_monitor_family) else "decoded_shape"
            monitor_meta = None
            if monitor_fn is not None and monitor_output_device is not None:
                if monitor_mode == "project":
                    project_wav = run_dir / "project_imitation.wav"
                    monitor_pcm, monitor_meta = _generate_project_imitation_audio(
                        family=resolved_monitor_family,
                        heldout_family="diphthong",
                        device_profile="highpass",
                        device_gain=1.0,
                        monitor_family=monitor_family,
                        noise_rms=0.0,
                        noise_seed=None,
                        duration_scale=1.0,
                        frame_size=frame_size,
                        output_wav=project_wav,
                    )
                    playback_meta = _play_pcm_on_device(monitor_pcm, output_device=monitor_output_device, label="project")
                    monitor_meta = {
                        **monitor_meta,
                        **playback_meta,
                        "resolved_family": resolved_monitor_family,
                        "phrase_family": phrase_monitor_family,
                        "decoded_family": decoded_monitor_family,
                        "phrase_token_count": phrase_token_count,
                        "family_source": monitor_family_source,
                    }
                elif monitor_mode == "decoded":
                    project_wav = run_dir / "project_imitation.wav"
                    vowel_segments = decoded.get("vowel_segments", [])
                    if vowel_segments:
                        vowel_timing = [
                            (seg["vowel"], seg["duration_frames"] * frame_size / SR)
                            for seg in vowel_segments
                        ]
                        vowel_sequence = [seg["vowel"] for seg in vowel_segments]
                    else:
                        vowel_sequence = list(decoded["decoded_vowels"])
                        vowel_timing = None
                    tts_duration = float(len(target_pcm)) / SR if len(target_pcm) > 0 else 2.0
                    silence_timing = _extract_silence_timing(decoded, frame_size=frame_size)
                    monitor_pcm, monitor_meta = _generate_decoded_imitation_audio(
                        decoded_vowels=vowel_sequence,
                        vowel_timing=vowel_timing,
                        silence_timing=silence_timing,
                        tts_duration=tts_duration,
                        hold_duration=hold_duration,
                        device_profile="highpass",
                        device_gain=1.0,
                        noise_rms=0.0,
                        noise_seed=None,
                        frame_size=frame_size,
                        output_wav=project_wav,
                    )
                    playback_meta = _play_pcm_on_device(monitor_pcm, output_device=monitor_output_device, label="project")
                    monitor_meta = {
                        **monitor_meta,
                        **playback_meta,
                        "resolved_family": "decoded",
                        "family_source": "decoded_vowels",
                    }
                else:
                    monitor_pcm = captured_pcm if monitor_mode == "captured" else target_pcm
                    monitor_meta = monitor_fn(pcm=monitor_pcm, output_device=monitor_output_device)
            passed = transition_count >= threshold_transitions and continuity >= threshold_continuity
            runs.append(
                {
                    "repeat": repeat_idx + 1,
                    "phrase_index": phrase_idx,
                    "phrase": phrase,
                    "tts_wav": str(tts_wav),
                    "captured_wav": str(captured_wav),
                    "tts_sample_count": int(len(target_pcm)),
                    "captured_sample_count": int(captured_meta["sample_count"]),
                    "capture": capture_meta,
                    "monitor": monitor_meta,
                    "decoded": {
                        "decoded_transition_count": transition_count,
                        "decoded_unique_vowel_count": int(decoded["decoded_unique_vowel_count"]),
                        "continuity": continuity,
                        "mean_boundary_delta": float(decoded["mean_boundary_delta"]),
                        "max_boundary_delta": float(decoded["max_boundary_delta"]),
                        "decoded_vowels_preview": list(decoded["decoded_vowels"][:20]),
                        "vowel_segments": decoded.get("vowel_segments", []),
                        "silence_threshold": decoded.get("silence_threshold", 0.0),
                    },
                    "verdict": {
                        "threshold_transitions": int(threshold_transitions),
                        "threshold_continuity": float(threshold_continuity),
                        "passed": passed,
                    },
                }
            )

    pass_count = sum(1 for run in runs if run["verdict"]["passed"])
    avg_transitions = float(np.mean([run["decoded"]["decoded_transition_count"] for run in runs])) if runs else 0.0
    avg_continuity = float(np.mean([run["decoded"]["continuity"] for run in runs])) if runs else 0.0
    failure_rows = [
        {
            "repeat": run["repeat"],
            "phrase_index": run["phrase_index"],
            "phrase": run["phrase"],
            "decoded_transition_count": run["decoded"]["decoded_transition_count"],
            "continuity": run["decoded"]["continuity"],
        }
        for run in runs
        if not run["verdict"]["passed"]
    ]
    summary = {
        "pass_count": pass_count,
        "total": len(runs),
        "pass_rate": float(pass_count / len(runs)) if runs else 0.0,
        "avg_transition_count": avg_transitions,
        "avg_continuity": avg_continuity,
        "threshold_transitions": int(threshold_transitions),
        "threshold_continuity": float(threshold_continuity),
        "verdict": pass_count == len(runs) if runs else False,
        "failure_rows": failure_rows,
    }
    report = {
        "output_dir": str(output_dir),
        "output": str(output_path),
        "phrases": phrase_list,
        "repeats": repeats,
        "input_device": int(input_device),
        "output_device": int(output_device),
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "lead_in_s": float(lead_in_s),
        "tail_s": float(tail_s),
        "dry_run": bool(dry_run),
        "monitor_mode": monitor_mode,
        "monitor_family": monitor_family,
        "monitor_output_device": int(monitor_output_device) if monitor_output_device is not None else None,
        "runs": runs,
        "summary": summary,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 TTS vowel-transition speaker harness")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/m9-tts-vowel-harness"))
    parser.add_argument("--output", type=Path, default=None, help="Optional report JSON path (defaults to output-dir/report.json)")
    parser.add_argument("--phrases", nargs="*", default=DEFAULT_PHRASES, help="Phrases to synthesize and play")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-device", type=int, default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--output-device", type=int, default=DEFAULT_OUTPUT_DEVICE)
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE)
    parser.add_argument("--rate", type=str, default=DEFAULT_RATE)
    parser.add_argument("--pitch", type=str, default=DEFAULT_PITCH)
    parser.add_argument("--lead-in-s", type=float, default=DEFAULT_LEAD_IN_S)
    parser.add_argument("--tail-s", type=float, default=DEFAULT_TAIL_S)
    parser.add_argument("--threshold-transitions", type=int, default=DEFAULT_MIN_TRANSITIONS)
    parser.add_argument("--threshold-continuity", type=float, default=DEFAULT_MIN_CONTINUITY)
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware capture and treat the synthesized phrase as perfectly captured")
    parser.add_argument("--monitor-output-device", type=int, default=DEFAULT_MONITOR_OUTPUT_DEVICE, help="Optional output device used to replay the project-generated imitation")
    parser.add_argument("--monitor-mode", choices=["captured", "target", "project", "decoded", "off"], default=DEFAULT_MONITOR_MODE, help="What audio to replay on the monitor output device")
    parser.add_argument("--monitor-family", type=str, default=DEFAULT_MONITOR_FAMILY, choices=["auto", *VOWEL_FAMILIES.keys()], help="Family rendered for the project-generated monitor playback")
    parser.add_argument("--hold-duration", type=float, default=0.0, help="Hold each vowel for this duration in seconds (0 = use TTS timing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tts_vowel_transition_harness(
        output_dir=args.output_dir,
        output=args.output,
        phrases=args.phrases,
        repeats=args.repeats,
        input_device=args.input_device,
        output_device=args.output_device,
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        lead_in_s=args.lead_in_s,
        tail_s=args.tail_s,
        threshold_transitions=args.threshold_transitions,
        threshold_continuity=args.threshold_continuity,
        frame_size=args.frame_size,
        dry_run=args.dry_run,
        monitor_output_device=args.monitor_output_device,
        monitor_mode=args.monitor_mode,
        monitor_family=args.monitor_family,
        hold_duration=args.hold_duration,
    )

    for run in report["runs"]:
        verdict = "PASS" if run["verdict"]["passed"] else "FAIL"
        decoded = run["decoded"]
        print(
            f"[repeat {run['repeat']:02d} phrase {run['phrase_index']:02d}] {verdict} "
            f"transitions={decoded['decoded_transition_count']} continuity={decoded['continuity']:.3f} "
            f"phrase={run['phrase']}"
        )
    summary = report["summary"]
    print(
        f"Summary: {summary['pass_count']}/{summary['total']} pass, "
        f"avg_transitions={summary['avg_transition_count']:.2f}, "
        f"avg_continuity={summary['avg_continuity']:.3f}"
    )
    print(f"Report: {report['output']}")


if __name__ == "__main__":
    main()
