"""Simulation-only signal-level self-suppression.

This module is the scoped bridge toward og-cora's signal-level corollary
discharge without importing its live playback assumptions. It models:

    efference PCM -> predicted self PCM -> observed mixture -> residual -> SVSR

No device I/O, no live playback, and no biological overclaim. The forward model
is deterministic room-coloring (delay + attenuation + echo), not learned. The
subtraction is signal-level PCM subtraction, intentionally separate from the
scheduling-level `self_suppressed` flag in theta_gamma.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .motor_plan import Vowel, synthesize_vowel


@dataclass(frozen=True)
class SelfSuppressionParams:
    sample_rate: int = 16000
    duration_ms: int = 300
    delay_ms: float = 10.0
    attenuation: float = 0.65
    echo_decay: float = 0.18
    world_gain: float = 0.0
    svsr_pass_db: float = 20.0


def _as_float32(pcm: np.ndarray) -> np.ndarray:
    return np.asarray(pcm, dtype=np.float32).reshape(-1)


def _pad_or_trim(pcm: np.ndarray, n: int) -> np.ndarray:
    pcm = _as_float32(pcm)
    if len(pcm) == n:
        return pcm
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, len(pcm))] = pcm[:n]
    return out


def predict_self_pcm(
    efference_pcm: np.ndarray,
    *,
    sample_rate: int = 16000,
    delay_ms: float = 10.0,
    attenuation: float = 0.65,
    echo_decay: float = 0.18,
) -> np.ndarray:
    """Deterministically predict room-colored self PCM from efference PCM.

    This is a toy forward model: delayed direct path + weaker single echo. It is
    deliberately simple, deterministic, and hardware-free; it gives the thalamus
    harness a real signal to subtract without requiring speaker playback.
    """
    eff = _as_float32(efference_pcm)
    out = np.zeros_like(eff, dtype=np.float32)
    delay = max(0, int(round(delay_ms * sample_rate / 1000.0)))
    echo_delay = max(1, int(round(15.0 * sample_rate / 1000.0)))

    if delay < len(out):
        out[delay:] += attenuation * eff[: len(out) - delay]
    echo_start = delay + echo_delay
    if echo_start < len(out):
        out[echo_start:] += attenuation * echo_decay * eff[: len(out) - echo_start]

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def mix_observed(
    self_pcm: np.ndarray | None,
    world_pcm: np.ndarray | None,
    *,
    world_gain: float = 1.0,
) -> np.ndarray:
    """Create an observed microphone mixture from predicted self + world PCM."""
    if self_pcm is None and world_pcm is None:
        raise ValueError("at least one of self_pcm or world_pcm is required")
    n = max(len(self_pcm) if self_pcm is not None else 0, len(world_pcm) if world_pcm is not None else 0)
    observed = np.zeros(n, dtype=np.float32)
    if self_pcm is not None:
        observed += _pad_or_trim(self_pcm, n)
    if world_pcm is not None:
        observed += float(world_gain) * _pad_or_trim(world_pcm, n)
    return np.clip(observed, -1.0, 1.0).astype(np.float32)


def spectral_subtract(
    observed_pcm: np.ndarray,
    predicted_self_pcm: np.ndarray,
    *,
    floor: float = 0.0,
) -> np.ndarray:
    """Signal-level subtraction gate over PCM.

    Name mirrors og-cora's Thalamus gate, but the simulation stays in PCM instead
    of mel bins to avoid scipy/librosa dependencies in cora-infans. With an exact
    prediction, self energy is removed; world-only signal is preserved when the
    prediction is zero.
    """
    obs = _as_float32(observed_pcm)
    pred = _pad_or_trim(predicted_self_pcm, len(obs))
    residual = obs - pred
    if floor > 0:
        residual = np.where(np.abs(residual) < floor * np.abs(obs), 0.0, residual)
    return np.clip(residual, -1.0, 1.0).astype(np.float32)


def rms(pcm: np.ndarray) -> float:
    x = _as_float32(pcm)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def compute_svsr_db(observed_pcm: np.ndarray, residual_pcm: np.ndarray) -> float:
    """Self-Voice Suppression Ratio: observed energy / residual energy in dB."""
    obs_e = float(np.sum(_as_float32(observed_pcm) ** 2))
    res_e = float(np.sum(_as_float32(residual_pcm) ** 2))
    if obs_e <= 1e-30:
        return 0.0
    if res_e <= 1e-30:
        return float("inf")
    return float(10.0 * math.log10(obs_e / res_e))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = _as_float32(a)
    bb = _pad_or_trim(b, len(aa))
    if len(aa) == 0 or float(np.std(aa)) <= 1e-9 or float(np.std(bb)) <= 1e-9:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def evaluate_forward_model(
    efference_pcm: np.ndarray,
    observed_pcm: np.ndarray,
    *,
    sample_rate: int = 16000,
    delay_ms: float = 10.0,
    attenuation: float = 0.65,
    echo_decay: float = 0.18,
    world_pcm: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one forward-model parameter setting against observed audio."""
    predicted = predict_self_pcm(
        efference_pcm,
        sample_rate=sample_rate,
        delay_ms=delay_ms,
        attenuation=attenuation,
        echo_decay=echo_decay,
    )
    observed = _as_float32(observed_pcm)
    residual = spectral_subtract(observed, predicted)
    report: dict[str, object] = {
        "params": {
            "delay_ms": float(delay_ms),
            "attenuation": float(attenuation),
            "echo_decay": float(echo_decay),
        },
        "observed_rms": rms(observed),
        "predicted_rms": rms(predicted),
        "residual_rms": rms(residual),
        "svsr_db": compute_svsr_db(observed, residual),
    }
    if world_pcm is not None:
        report["world_residual_correlation"] = _corr(world_pcm, residual)
    return report


def fit_forward_model(
    efference_pcm: np.ndarray,
    observed_pcm: np.ndarray,
    *,
    sample_rate: int = 16000,
    delay_grid_ms: tuple[float, ...] = (4.0, 8.0, 12.0, 18.0, 24.0),
    attenuation_grid: tuple[float, ...] = (0.25, 0.45, 0.55, 0.65, 0.8),
    echo_decay_grid: tuple[float, ...] = (0.05, 0.12, 0.22, 0.35),
    world_pcm: np.ndarray | None = None,
    svsr_pass_db: float = 20.0,
) -> dict[str, object]:
    """Fit the toy forward model by deterministic grid search.

    This is deliberately transparent cognitive learning: choose the delay / gain
    / echo params that minimize prediction residual (equivalently maximize SVSR)
    for a known efference->observation pair. No live playback and no learned
    black box.
    """
    best: dict[str, object] | None = None
    for delay in delay_grid_ms:
        for attenuation in attenuation_grid:
            for echo_decay in echo_decay_grid:
                cand = evaluate_forward_model(
                    efference_pcm,
                    observed_pcm,
                    sample_rate=sample_rate,
                    delay_ms=delay,
                    attenuation=attenuation,
                    echo_decay=echo_decay,
                    world_pcm=world_pcm,
                )
                if best is None:
                    best = cand
                    continue
                # Primary: residual RMS. Tie-breaker: higher SVSR.
                cand_key = (float(cand["residual_rms"]), -float(cand["svsr_db"]))
                best_key = (float(best["residual_rms"]), -float(best["svsr_db"]))
                if cand_key < best_key:
                    best = cand
    assert best is not None
    learned = dict(best["params"])  # type: ignore[arg-type]
    report = dict(best)
    report["learned_params"] = learned
    report["verdict"] = (
        "learned_forward_model"
        if float(report["svsr_db"]) >= svsr_pass_db
        else "partial_learning"
        if float(report["residual_rms"]) < float(report["observed_rms"])
        else "not_learned"
    )
    return report


def run_forward_model_learning_trial(
    *,
    vowel: Vowel = "a",
    world_vowel: Vowel | None = None,
    sample_rate: int = 16000,
    duration_ms: int = 300,
    observed_delay_ms: float = 18.0,
    observed_attenuation: float = 0.55,
    observed_echo_decay: float = 0.22,
    world_gain: float = 0.0,
    svsr_pass_db: float = 20.0,
) -> dict[str, object]:
    """Simulate learning a forward model from prediction error.

    Hidden observed parameters create the "room"; a bad initial guess is scored;
    then a deterministic grid search chooses parameters that reduce residual
    prediction error. This is the cognitive learning step, still simulation-only.
    """
    efference = synthesize_vowel(vowel, duration_ms=duration_ms, sample_rate=sample_rate)
    observed_self = predict_self_pcm(
        efference,
        sample_rate=sample_rate,
        delay_ms=observed_delay_ms,
        attenuation=observed_attenuation,
        echo_decay=observed_echo_decay,
    )
    world = (
        synthesize_vowel(world_vowel, duration_ms=duration_ms, sample_rate=sample_rate)
        if world_vowel is not None
        else None
    )
    observed = mix_observed(
        observed_self,
        world,
        world_gain=world_gain if world is not None else 0.0,
    )
    initial = evaluate_forward_model(
        efference,
        observed,
        sample_rate=sample_rate,
        delay_ms=4.0,
        attenuation=0.2,
        echo_decay=0.05,
        world_pcm=world,
    )
    fit = fit_forward_model(
        efference,
        observed,
        sample_rate=sample_rate,
        delay_grid_ms=(4.0, 8.0, 12.0, observed_delay_ms, 24.0),
        attenuation_grid=(0.2, 0.35, observed_attenuation, 0.75),
        echo_decay_grid=(0.05, 0.12, observed_echo_decay, 0.35),
        world_pcm=world,
        svsr_pass_db=svsr_pass_db,
    )
    initial_svsr = float(initial["svsr_db"])
    learned_svsr = float(fit["svsr_db"])
    improvement = learned_svsr - initial_svsr
    verdict = (
        "learned_forward_model"
        if learned_svsr >= svsr_pass_db and improvement > 3.0
        else "partial_learning"
        if improvement > 0.0
        else "not_learned"
    )
    return {
        "mode": "simulation_only_forward_model_learning",
        "live_playback": False,
        "sample_rate": sample_rate,
        "duration_ms": duration_ms,
        "vowel": vowel,
        "world_vowel": world_vowel,
        "world_gain": world_gain if world is not None else 0.0,
        "true_params": {
            "delay_ms": observed_delay_ms,
            "attenuation": observed_attenuation,
            "echo_decay": observed_echo_decay,
        },
        "initial_params": initial["params"],
        "learned_params": fit["learned_params"],
        "initial_svsr_db": initial_svsr,
        "learned_svsr_db": learned_svsr,
        "svsr_improvement_db": improvement,
        "initial_residual_rms": initial["residual_rms"],
        "learned_residual_rms": fit["residual_rms"],
        "world_residual_correlation": fit.get("world_residual_correlation"),
        "verdict": verdict,
    }



def run_self_suppression_trial(
    *,
    vowel: Vowel = "a",
    world_vowel: Vowel | None = None,
    params: SelfSuppressionParams | None = None,
) -> dict[str, object]:
    """Run one deterministic signal-level self-suppression simulation."""
    cfg = params or SelfSuppressionParams()
    efference = synthesize_vowel(vowel, duration_ms=cfg.duration_ms, sample_rate=cfg.sample_rate)
    predicted = predict_self_pcm(
        efference,
        sample_rate=cfg.sample_rate,
        delay_ms=cfg.delay_ms,
        attenuation=cfg.attenuation,
        echo_decay=cfg.echo_decay,
    )
    world = (
        synthesize_vowel(world_vowel, duration_ms=cfg.duration_ms, sample_rate=cfg.sample_rate)
        if world_vowel is not None
        else None
    )
    observed = mix_observed(predicted, world, world_gain=cfg.world_gain if world is not None else 0.0)
    residual = spectral_subtract(observed, predicted)
    svsr = compute_svsr_db(observed, residual)
    residual_rms = rms(residual)
    observed_rms = rms(observed)
    if svsr >= cfg.svsr_pass_db:
        verdict = "self_suppressed"
    elif residual_rms < observed_rms:
        verdict = "partial_suppression"
    else:
        verdict = "not_suppressed"

    return {
        "mode": "simulation_only_signal_self_suppression",
        "live_playback": False,
        "sample_rate": cfg.sample_rate,
        "duration_ms": cfg.duration_ms,
        "vowel": vowel,
        "world_vowel": world_vowel,
        "world_gain": cfg.world_gain if world is not None else 0.0,
        "observed_rms": observed_rms,
        "residual_rms": residual_rms,
        "svsr_db": svsr,
        "svsr_pass_db": cfg.svsr_pass_db,
        "verdict": verdict,
    }
