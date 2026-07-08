"""Simulation-only signal-level self-suppression harness.

This is the scoped bridge toward og-cora's signal-level corollary discharge:
real efference PCM -> predicted self -> subtraction -> SVSR metric, but with
NO live playback, NO device access, and NO biological overclaim.
"""

from __future__ import annotations

import numpy as np

from cora_infans.io.motor_plan import synthesize_vowel
from cora_infans.io.self_suppression import (
    SelfSuppressionParams,
    compute_svsr_db,
    mix_observed,
    predict_self_pcm,
    run_self_suppression_trial,
    spectral_subtract,
)


def test_predict_self_pcm_adds_deterministic_room_coloring() -> None:
    efference = np.zeros(1600, dtype=np.float32)
    efference[100] = 1.0

    predicted = predict_self_pcm(
        efference,
        sample_rate=16000,
        delay_ms=10.0,
        attenuation=0.5,
        echo_decay=0.25,
    )

    assert predicted.dtype == np.float32
    assert len(predicted) == len(efference)
    # Direct path appears after the configured delay.
    assert predicted[260] == np.float32(0.5)  # 100 + 10ms * 16 samples/ms
    # Echo tail appears later and weaker.
    assert 0.0 < float(predicted[500]) < float(predicted[260])


def test_spectral_subtraction_reduces_self_residual_energy() -> None:
    efference = synthesize_vowel("a", duration_ms=300, sample_rate=16000)
    predicted = predict_self_pcm(efference, sample_rate=16000)
    observed = mix_observed(predicted, world_pcm=None)

    residual = spectral_subtract(observed, predicted)

    raw_energy = float(np.mean(observed**2))
    residual_energy = float(np.mean(residual**2))
    assert residual_energy < raw_energy * 0.05
    assert compute_svsr_db(observed, residual) > 20.0


def test_world_only_signal_is_not_erased_by_empty_prediction() -> None:
    world = synthesize_vowel("e", duration_ms=300, sample_rate=16000)
    predicted = np.zeros_like(world)
    observed = mix_observed(self_pcm=None, world_pcm=world)

    residual = spectral_subtract(observed, predicted)

    # With no self prediction, the residual should preserve the world signal.
    assert np.corrcoef(world, residual)[0, 1] > 0.98
    assert compute_svsr_db(observed, residual) < 1.0


def test_mixed_self_and_world_leaves_world_residual() -> None:
    self_pcm = predict_self_pcm(
        synthesize_vowel("a", duration_ms=300, sample_rate=16000), sample_rate=16000
    )
    world = synthesize_vowel("o", duration_ms=300, sample_rate=16000)
    observed = mix_observed(self_pcm=self_pcm, world_pcm=world, world_gain=0.35)

    residual = spectral_subtract(observed, self_pcm)

    assert compute_svsr_db(observed, residual) > 3.0
    # The residual should still correlate with the world component.
    assert np.corrcoef(world, residual)[0, 1] > 0.25


def test_run_self_suppression_trial_reports_signal_level_metric() -> None:
    report = run_self_suppression_trial(
        vowel="a",
        world_vowel="o",
        params=SelfSuppressionParams(duration_ms=300, world_gain=0.25),
    )

    assert report["mode"] == "simulation_only_signal_self_suppression"
    assert report["live_playback"] is False
    assert report["sample_rate"] == 16000
    assert report["vowel"] == "a"
    assert report["world_vowel"] == "o"
    assert report["svsr_db"] > 3.0
    assert report["residual_rms"] < report["observed_rms"]
    assert report["verdict"] in ("self_suppressed", "partial_suppression")
