"""Tests for M9b convergence helpers."""

from __future__ import annotations

import numpy as np

from m9_sound.convergence import (
    compute_repeatability_stats,
    refine_prototype,
    summarize_audio,
)
from sag.acoustic_models import AcousticNode


class TestSummarizeAudio:
    def test_summarize_basic_tone(self):
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        summary = summarize_audio(audio, sr)
        assert summary["duration_s"] == 1.0
        assert summary["rms"] > 0.0
        assert summary["loudness_db"] < 0.0
        assert summary["pitch_hz"] > 0.0
        assert len(summary["mfcc_centroid"]) == 13


class TestRepeatabilityStats:
    def test_compute_stats(self):
        stats = compute_repeatability_stats([0.5, 0.52, 0.51, 0.49, 0.5])
        assert stats.n_trials == 5
        assert stats.mean > 0.0
        assert stats.std >= 0.0
        assert stats.cv < 0.2
        assert stats.converged is True

    def test_empty_scores(self):
        stats = compute_repeatability_stats([])
        assert stats.n_trials == 0
        assert stats.mean == 0.0
        assert stats.converged is False


class TestRefinePrototype:
    def test_refine_updates_history_and_summary(self):
        proto = AcousticNode.from_audio_summary(
            id="proto_440_sine",
            core="440 Hz sine target prototype",
            grounding=["440hz", "sine"],
            label="proto_440_sine",
            pitch_mean_hz=440.0,
            duration_mean_s=1.0,
            loudness_mean_db=-12.0,
            mfcc_centroid=[0.1] * 13,
            source_trials=["target"],
        )
        updated = refine_prototype(
            proto,
            trial_id="trial_001",
            similarity=0.75,
            audio_summary={
                "pitch_hz": 441.0,
                "duration_s": 1.0,
                "loudness_db": -11.5,
                "mfcc_centroid": [0.2] * 13,
            },
        )
        assert updated.production_count == 2
        assert updated.avg_similarity == 0.75
        assert updated.pitch_mean_hz == 441.0
        assert updated.duration_mean_s == 1.0
        assert updated.loudness_mean_db == -11.5
        assert updated.source_trials == ["target", "trial_001"]
        assert len(updated.similarity_history) == 1
