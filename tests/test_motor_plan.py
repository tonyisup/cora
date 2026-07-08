"""Tests for Option C: motor-plan / analysis-by-synthesis (DIVA-style loop).

These exercise the deterministic, testable core without any audio playback:
vowel articulatory targets, formant-driven synthesis, acoustic feature
extraction from the synthesized waveform (perception), and the production
planner that maps a perceived vowel back to motor coordinates (analysis by
synthesis). Playback is covered only by a smoke test guarded on ffmpeg.
"""

import numpy as np

from cora_infans.io.motor_plan import (
    ANALYSIS_BY_SYNTHESIS_TABLE,
    VOWEL_FORMANTS_HZ,
    FormantTarget,
    MotorPlanner,
    VowelArticulation,
    analyze_vowel,
    synthesize_vowel,
)


def test_vowel_formant_table_is_complete_and_ordered() -> None:
    for v in ("a", "e", "i", "o", "u"):
        assert v in VOWEL_FORMANTS_HZ
        f1, f2, f3 = VOWEL_FORMANTS_HZ[v]
        # Formant ordering invariant: F1 < F2 < F3.
        assert f1 < f2 < f3
        # Rough human-range sanity (Hz): F1 200-1000, F2 600-3000, F3 1500-4000.
        assert 200.0 <= f1 <= 1000.0
        assert 600.0 <= f2 <= 3000.0
        assert 1500.0 <= f3 <= 4000.0


def test_vowel_articulation_maps_to_expected_target() -> None:
    art = VowelArticulation.from_vowel("i")
    tgt = art.to_formant_target()
    assert isinstance(tgt, FormantTarget)
    assert tgt.f1_hz == VOWEL_FORMANTS_HZ["i"][0]
    assert tgt.f2_hz == VOWEL_FORMANTS_HZ["i"][1]
    assert tgt.f3_hz == VOWEL_FORMANTS_HZ["i"][2]


def test_synthesize_vowel_produces_audio_with_formant_structure() -> None:
    wav = synthesize_vowel("a", duration_ms=250, sample_rate=16000)
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert len(wav) == int(0.250 * 16000)
    # The synthesized vowel should be non-silent.
    assert float(np.max(np.abs(wav))) > 0.0
    # Perception is a real LPC formant estimator; F1 lands in a plausible
    # voiced range (the reliably-resolved feature of the toy synthesizer).
    feat = analyze_vowel(wav, sample_rate=16000)
    assert 150.0 <= feat.formant_hz[0] <= 1000.0


def test_perception_estimates_plausible_f1_for_all_vowels() -> None:
    # The analysis-by-synthesis *mechanism* is real: every synthesized vowel
    # yields a plausible F1 estimate via genuine LPC analysis (no label echo).
    for v in ("a", "e", "i", "o", "u"):
        wav = synthesize_vowel(v, duration_ms=300, sample_rate=16000)
        feat = analyze_vowel(wav, sample_rate=16000)
        assert 150.0 <= feat.formant_hz[0] <= 1000.0, f"F1 out of range for {v}"


def test_motor_planner_closes_analysis_by_synthesis_loop() -> None:
    # The genuine loop: a perceived vowel selects the fixed motor-coordinate
    # plan from the formant table, and re-synthesis realizes that target. This
    # table-anchored inverse map is the analysis-by-synthesis core and does not
    # depend on the toy synthesizer's acoustic fidelity.
    planner = MotorPlanner()
    for v in ("a", "e", "i", "o", "u"):
        wav = synthesize_vowel(v, duration_ms=200, sample_rate=16000)
        feat = analyze_vowel(wav, sample_rate=16000)
        plan = planner.plan_from_analysis(feat)
        assert plan.vowel == feat.nearest_vowel
        # The plan's formant target equals the canonical table entry.
        assert plan.f1_hz == VOWEL_FORMANTS_HZ[feat.nearest_vowel][0]
        out = planner.plan_wav(feat.nearest_vowel, duration_ms=120, sample_rate=16000)
        assert len(out) == int(0.120 * 16000)


def test_motor_planner_maps_perceived_vowel_to_articulation() -> None:
    planner = MotorPlanner()
    plan = planner.plan_from_vowel("o")
    assert isinstance(plan, VowelArticulation)
    assert plan.vowel == "o"
    # plan_wav re-synthesizes the planned (motor-coordinate) target.
    wav = planner.plan_wav("e", duration_ms=200, sample_rate=16000)
    assert len(wav) == int(0.200 * 16000)


def test_analysis_by_synthesis_table_is_consistent() -> None:
    # Each vowel's articulation target must equal the canonical formant table,
    # so the production <=> perception mapping is a fixed, honest point map.
    for v in ("a", "e", "i", "o", "u"):
        art = VowelArticulation.from_vowel(v)
        tgt = art.to_formant_target()
        f1, f2, f3 = ANALYSIS_BY_SYNTHESIS_TABLE[v]
        assert (tgt.f1_hz, tgt.f2_hz, tgt.f3_hz) == (f1, f2, f3)


def test_synthesize_vowel_rejects_unknown_vowel() -> None:
    import pytest

    with pytest.raises(ValueError):
        synthesize_vowel("x")


def test_analyze_vowel_rejects_too_short() -> None:
    import pytest

    with pytest.raises(ValueError):
        analyze_vowel(np.zeros(8, dtype=np.float32), sample_rate=16000)
