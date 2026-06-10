"""Regression tests for the M9 TTS vowel-transition speaker harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "m9_tts_vowel_transition_harness.py"
SPEC = importlib.util.spec_from_file_location("m9_tts_vowel_transition_harness", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)

SR = 16_000


def _fake_tts_audio_fn(*, text: str, output_wav: Path, voice: str, rate: str, pitch: str) -> np.ndarray:
    # Build a deterministic vowel-ish carrier so the harness has a real PCM
    # artifact to persist, but keep the test independent of macOS TTS.
    duration_s = 0.75 + 0.05 * len(text.split())
    sample_count = int(SR * duration_s)
    t = np.linspace(0.0, duration_s, sample_count, endpoint=False, dtype=np.float32)
    base_freq = 180.0 + 7.5 * len(text)
    pcm = 0.12 * np.sin(2.0 * np.pi * base_freq * t).astype(np.float32)
    HARNESS.write_pcm16_wav(output_wav, pcm, sr=SR)
    return pcm


def test_parse_phrases_uses_nonempty_values():
    phrases = HARNESS._parse_phrases(["  eee ahh  ", "ooo ohh", "", "   "])
    assert phrases == ["eee ahh", "ooo ohh"]


def test_tts_vowel_transition_harness_project_monitor_mode_plays_project_audio(tmp_path: Path):
    output_dir = tmp_path / "tts-harness-project"
    output = tmp_path / "project-report.json"
    playback_calls: list[dict[str, object]] = []

    def fake_project_imitation_audio(*, family: str, heldout_family: str, device_profile: str, device_gain: float, monitor_family: str, noise_rms: float, noise_seed: int | None, duration_scale: float, frame_size: int, output_wav: Path):
        pcm = np.full(SR // 4, 0.05, dtype=np.float32)
        HARNESS.write_pcm16_wav(output_wav, pcm, sr=SR)
        return pcm, {
            "family": family,
            "monitor_family": monitor_family,
            "heldout_family": heldout_family,
            "device_profile": device_profile,
            "device_gain": device_gain,
            "duration_scale": duration_scale,
            "sample_count": int(len(pcm)),
            "wav_path": str(output_wav),
            "decoded_transition_count": 3,
        }

    def fake_play_pcm_on_device(pcm: np.ndarray, *, output_device: int, label: str = "monitor") -> dict[str, object]:
        meta = {"output_device": output_device, "label": label, "sample_count": int(len(pcm))}
        playback_calls.append(meta)
        return meta

    original_generate = getattr(HARNESS, "_generate_project_imitation_audio")
    original_play = getattr(HARNESS, "_play_pcm_on_device")
    try:
        setattr(HARNESS, "_generate_project_imitation_audio", fake_project_imitation_audio)
        setattr(HARNESS, "_play_pcm_on_device", fake_play_pcm_on_device)
        report = HARNESS.run_tts_vowel_transition_harness(
            output_dir=output_dir,
            output=output,
            phrases=["eee ehh ahh ohh ooo"],
            repeats=1,
            dry_run=True,
            monitor_mode="project",
            monitor_family="phase_triad",
            tts_audio_fn=_fake_tts_audio_fn,
            threshold_transitions=0,
            threshold_continuity=0.0,
        )
    finally:
        setattr(HARNESS, "_generate_project_imitation_audio", original_generate)
        setattr(HARNESS, "_play_pcm_on_device", original_play)

    assert report["summary"]["verdict"] is True
    assert report["runs"][0]["monitor"]["label"] == "project"
    assert report["runs"][0]["monitor"]["output_device"] == 3
    assert report["runs"][0]["monitor"]["sample_count"] == SR // 4
    assert playback_calls == [{"output_device": 3, "label": "project", "sample_count": SR // 4}]
    assert (output_dir / "repeat-01" / "phrase-01" / "project_imitation.wav").exists()


def test_select_project_monitor_family_tracks_shape():
    assert HARNESS._select_project_monitor_family(4, 9) == "phase_triad"
    assert HARNESS._select_project_monitor_family(4, 4) == "phase_arc"
    assert HARNESS._select_project_monitor_family(8, 7) == "long_arc"


def test_tts_vowel_transition_harness_auto_monitor_family_uses_decoded_shape(tmp_path: Path):
    output_dir = tmp_path / "tts-harness-auto"
    output = tmp_path / "auto-report.json"
    observed: dict[str, object] = {}

    def fake_decode_vowel_transition_audio(audio: np.ndarray, *, frame_size: int, **kwargs: object) -> dict[str, object]:
        return {
            "decoded_vowels": ["i", "E", "a", "O", "u", "O", "a", "E"],
            "decoded_transition_count": 7,
            "decoded_unique_vowel_count": 8,
            "continuity": 0.95,
            "mean_boundary_delta": 0.005,
            "max_boundary_delta": 0.01,
            "min_boundary_delta": 0.0,
        }

    def fake_project_imitation_audio(*, family: str, heldout_family: str, device_profile: str, device_gain: float, monitor_family: str, noise_rms: float, noise_seed: int | None, duration_scale: float, frame_size: int, output_wav: Path):
        observed["family"] = family
        observed["heldout_family"] = heldout_family
        pcm = np.full(SR // 4, 0.05, dtype=np.float32)
        HARNESS.write_pcm16_wav(output_wav, pcm, sr=SR)
        return pcm, {
            "family": family,
            "monitor_family": monitor_family,
            "heldout_family": heldout_family,
            "device_profile": device_profile,
            "device_gain": device_gain,
            "duration_scale": duration_scale,
            "sample_count": int(len(pcm)),
            "wav_path": str(output_wav),
            "decoded_transition_count": 7,
        }

    def fake_play_pcm_on_device(pcm: np.ndarray, *, output_device: int, label: str = "monitor") -> dict[str, object]:
        return {"output_device": output_device, "label": label, "sample_count": int(len(pcm))}

    original_decode = getattr(HARNESS, "decode_vowel_transition_audio")
    original_generate = getattr(HARNESS, "_generate_project_imitation_audio")
    original_play = getattr(HARNESS, "_play_pcm_on_device")
    try:
        setattr(HARNESS, "decode_vowel_transition_audio", fake_decode_vowel_transition_audio)
        setattr(HARNESS, "_generate_project_imitation_audio", fake_project_imitation_audio)
        setattr(HARNESS, "_play_pcm_on_device", fake_play_pcm_on_device)
        report = HARNESS.run_tts_vowel_transition_harness(
            output_dir=output_dir,
            output=output,
            phrases=["eee ehh ahh ohh ooo"],
            repeats=1,
            dry_run=True,
            monitor_mode="project",
            monitor_family="auto",
            tts_audio_fn=_fake_tts_audio_fn,
            threshold_transitions=0,
            threshold_continuity=0.0,
        )
    finally:
        setattr(HARNESS, "decode_vowel_transition_audio", original_decode)
        setattr(HARNESS, "_generate_project_imitation_audio", original_generate)
        setattr(HARNESS, "_play_pcm_on_device", original_play)

    assert observed["family"] == "long_arc"
    assert report["monitor_family"] == "auto"
    assert report["runs"][0]["monitor"]["family"] == "long_arc"
    assert report["runs"][0]["monitor"]["resolved_family"] == "long_arc"
    assert report["runs"][0]["monitor"]["family_source"] == "decoded_shape"
