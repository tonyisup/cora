"""Tests for M10 consonant-aware decoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from m9_sound.vowel_transitions import (
    _decode_vowel_frames,
    decode_vowel_transition_audio,
    _synthesize_vowel_frames,
)


def _make_silence(n_samples: int = 4800) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


def _make_tone(freq: float = 440.0, duration_s: float = 0.3, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_consonant_gate_labels_far_frames_as_consonant():
    """Frames far from all vowel centroids should be labeled 'C' when consonant_gate is active."""
    tone = _make_tone(440.0, 0.3)
    # With a very low gate threshold, even moderate-distance frames should become C
    decoded = _decode_vowel_frames(tone, consonant_gate=0.5, silence_threshold=0.0)
    assert "C" in decoded, f"Expected 'C' in decoded labels, got: {set(decoded)}"


def test_consonant_gate_off_preserves_vowel_labels():
    """With consonant_gate=0 (default), frames should only be vowels or silence."""
    tone = _make_tone(440.0, 0.3)
    decoded = _decode_vowel_frames(tone, consonant_gate=0.0, silence_threshold=0.0)
    for label in decoded:
        assert label in ("i", "E", "a", "O", "u", "_"), f"Unexpected label: {label}"


def test_silence_frames_remain_silence():
    """Silence frames should always be labeled '_' regardless of consonant gate."""
    silence = _make_silence(4800)
    decoded = _decode_vowel_frames(silence, consonant_gate=2.0, silence_threshold=0.01)
    assert all(l == "_" for l in decoded), f"Expected all silence, got: {set(decoded)}"


def test_decode_report_includes_consonant_stats():
    """The decode report should include consonant_count and consonant_rate."""
    tone = _make_tone(440.0, 0.3)
    report = decode_vowel_transition_audio(tone, consonant_gate=0.5, silence_threshold=0.0)
    assert "consonant_count" in report, "Missing consonant_count in report"
    assert "consonant_rate" in report, "Missing consonant_rate in report"
    assert isinstance(report["consonant_count"], int)
    assert isinstance(report["consonant_rate"], float)


def test_decode_report_consonant_count_matches_labels():
    """consonant_count should match the number of 'C' labels in decoded_vowels."""
    tone = _make_tone(440.0, 0.3)
    report = decode_vowel_transition_audio(tone, consonant_gate=0.5, silence_threshold=0.0)
    c_count = sum(1 for l in report["decoded_vowels"] if l == "C")
    assert report["consonant_count"] == c_count, (
        f"consonant_count={report['consonant_count']} != actual C count={c_count}"
    )


def test_vowel_dominant_audio_has_few_consonants_at_default_gate():
    """With consonant_gate=0 (default), vowel audio should have zero consonant frames."""
    stimulus, _ = _synthesize_vowel_frames(
        ["i", "E", "a"],
        duration_s=0.6,
        sr=16000,
        frame_size=480,
    )
    # With gate off, no frames should be labeled C
    report = decode_vowel_transition_audio(
        stimulus, consonant_gate=0.0, silence_threshold=0.0
    )
    assert report["consonant_count"] == 0, (
        f"Expected 0 consonants with gate off, got {report['consonant_count']}"
    )
