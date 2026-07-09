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
    evaluate_forward_model,
    fit_forward_model,
    mix_observed,
    predict_self_pcm,
    run_forward_model_learning_trial,
    run_forward_model_robustness_sweep,
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


def test_fit_forward_model_recovers_hidden_room_coloring_params() -> None:
    efference = synthesize_vowel("a", duration_ms=300, sample_rate=16000)
    observed = predict_self_pcm(
        efference,
        sample_rate=16000,
        delay_ms=18.0,
        attenuation=0.55,
        echo_decay=0.22,
    )

    fit = fit_forward_model(
        efference,
        observed,
        sample_rate=16000,
        delay_grid_ms=(8.0, 14.0, 18.0, 24.0),
        attenuation_grid=(0.35, 0.55, 0.75),
        echo_decay_grid=(0.08, 0.22, 0.35),
    )

    learned = fit["learned_params"]
    assert learned["delay_ms"] == 18.0
    assert learned["attenuation"] == 0.55
    assert learned["echo_decay"] == 0.22
    assert fit["svsr_db"] > 30.0


def test_forward_model_learning_improves_svsr_over_bad_initial_guess() -> None:
    efference = synthesize_vowel("a", duration_ms=300, sample_rate=16000)
    observed = predict_self_pcm(
        efference,
        sample_rate=16000,
        delay_ms=18.0,
        attenuation=0.55,
        echo_decay=0.22,
    )
    initial = evaluate_forward_model(
        efference,
        observed,
        sample_rate=16000,
        delay_ms=4.0,
        attenuation=0.2,
        echo_decay=0.05,
    )
    fit = fit_forward_model(
        efference,
        observed,
        sample_rate=16000,
        delay_grid_ms=(4.0, 10.0, 18.0),
        attenuation_grid=(0.2, 0.55),
        echo_decay_grid=(0.05, 0.22),
    )

    assert fit["svsr_db"] > initial["svsr_db"] + 15.0
    assert fit["residual_rms"] < initial["residual_rms"] * 0.2


def test_wrong_efference_does_not_fake_good_forward_model_learning() -> None:
    observed_self = predict_self_pcm(
        synthesize_vowel("a", duration_ms=300, sample_rate=16000),
        sample_rate=16000,
        delay_ms=18.0,
        attenuation=0.55,
        echo_decay=0.22,
    )
    correct_fit = fit_forward_model(
        synthesize_vowel("a", duration_ms=300, sample_rate=16000),
        observed_self,
        sample_rate=16000,
        delay_grid_ms=(10.0, 18.0),
        attenuation_grid=(0.35, 0.55),
        echo_decay_grid=(0.08, 0.22),
    )
    wrong_fit = fit_forward_model(
        synthesize_vowel("i", duration_ms=300, sample_rate=16000),
        observed_self,
        sample_rate=16000,
        delay_grid_ms=(10.0, 18.0),
        attenuation_grid=(0.35, 0.55),
        echo_decay_grid=(0.08, 0.22),
    )

    assert correct_fit["svsr_db"] > wrong_fit["svsr_db"] + 10.0
    assert wrong_fit["verdict"] != "learned_forward_model"


def test_forward_model_learning_reports_world_preservation_metric() -> None:
    report = run_forward_model_learning_trial(
        vowel="a",
        world_vowel="o",
        observed_delay_ms=18.0,
        observed_attenuation=0.55,
        observed_echo_decay=0.22,
        world_gain=0.25,
    )

    assert report["mode"] == "simulation_only_forward_model_learning"
    assert report["live_playback"] is False
    assert report["learned_svsr_db"] > report["initial_svsr_db"]
    assert report["svsr_improvement_db"] > 3.0
    assert report["world_residual_correlation"] > 0.2
    assert report["verdict"] in ("learned_forward_model", "partial_learning")


def test_forward_model_robustness_sweep_reports_delay_degradation_and_world_preservation() -> None:
    sweep = run_forward_model_robustness_sweep(
        vowel="a",
        world_vowel="o",
        delay_errors_ms=(-12.0, 0.0, 12.0),
        attenuation_scales=(1.0,),
        echo_decay_scales=(1.0,),
        world_gains=(0.0, 0.25),
    )

    assert sweep["mode"] == "simulation_only_forward_model_robustness_sweep"
    assert sweep["live_playback"] is False
    assert sweep["case_count"] == 6
    exact_cases = [case for case in sweep["cases"] if case["delay_error_ms"] == 0.0 and case["world_gain"] == 0.0]
    delayed_cases = [case for case in sweep["cases"] if abs(case["delay_error_ms"]) == 12.0 and case["world_gain"] == 0.0]
    assert exact_cases[0]["svsr_db"] > 20.0
    assert max(case["svsr_db"] for case in delayed_cases) < exact_cases[0]["svsr_db"] - 10.0
    world_cases = [case for case in sweep["cases"] if case["world_gain"] == 0.25]
    assert all(case["world_residual_correlation"] is not None for case in world_cases)
    assert sweep["summary"]["best_case"]["delay_error_ms"] == 0.0
    assert sweep["summary"]["best_case"]["world_gain"] == 0.0
    assert sweep["summary"]["max_world_residual_correlation"] > 0.2
    assert sweep["summary"]["worst_svsr_db"] < sweep["summary"]["best_svsr_db"]
