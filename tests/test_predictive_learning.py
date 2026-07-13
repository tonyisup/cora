from __future__ import annotations

import pytest

from cora_infans.io.offline_eval import Annotation, CorpusClip, FrameObservation
from cora_infans.predictive_learning import (
    LocalPredictiveHierarchy,
    evaluate_online_predictor_safety,
    evaluate_predictive_corpus,
    evaluate_predictor_sequence,
)


def test_local_hierarchy_has_bounded_state_and_top_down_feedback() -> None:
    hierarchy = LocalPredictiveHierarchy()

    assert hierarchy.update(100.0) is None
    first = hierarchy.update(110.0)
    second = hierarchy.update(120.0)

    assert hierarchy.state_size == 10
    assert first is not None and first.top_down_correction == 0.0
    assert second is not None and second.top_down_correction != 0.0


def test_local_hierarchy_is_compared_against_transparent_baselines() -> None:
    report = evaluate_predictor_sequence(
        [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    )

    assert report["observation_count"] == 6
    assert report["bounded_state_scalars"] == 10
    assert set(report["baselines"]) == {
        "persistence",
        "exponential_moving_average",
    }
    assert report["local_predictive_hierarchy"]["count"] == 5
    assert len(report["steps"]) == 5
    assert set(report["steps"][-1]["expert_weights"]) == {
        "persistence",
        "exponential_moving_average",
        "top_down_trend",
    }


def test_local_hierarchy_scales_corrections_for_low_amplitude_features() -> None:
    report = evaluate_predictor_sequence([0.01, 0.012, 0.009, 0.011, 0.010])

    predictions = [step["prediction"] for step in report["steps"]]
    assert all(0.0 < prediction < 0.03 for prediction in predictions)


def test_local_hierarchy_rejects_non_finite_observations() -> None:
    hierarchy = LocalPredictiveHierarchy()

    with pytest.raises(ValueError, match="finite"):
        hierarchy.update(float("nan"))


def test_predictive_corpus_uses_held_out_feature_observations(tmp_path) -> None:
    dev = CorpusClip(
        "dev",
        tmp_path / "dev.wav",
        (Annotation(0.0, 1.0, "speech"),),
        "dev",
    )
    held = CorpusClip(
        "held",
        tmp_path / "held.wav",
        (Annotation(0.0, 1.0, "speech"),),
        "held_out",
    )

    def frames_for(clip, frame_ms):
        base = 100.0 if clip.clip_id == "dev" else 200.0
        return [
            FrameObservation(
                timestamp_ms=index * frame_ms,
                channel_energies=[0.1],
                features={"dominant_frequency_hz": base + index * 10.0},
                source="pcm_wav",
            )
            for index in range(6)
        ]

    report = evaluate_predictive_corpus([dev, held], frames_for)

    assert report["evaluated_clip_count"] == 1
    assert report["reports"][0]["clip_id"] == "held"
    assert report["reports"][0]["observation_count"] == 6
    assert "phase_template_top_down" in report["reports"][0]
    assert report["learned_phase_count"] == 6


def test_online_safety_report_separates_regret_from_superiority(tmp_path) -> None:
    clips = [
        CorpusClip(
            "clip",
            tmp_path / "clip.wav",
            (Annotation(0.0, 1.0, "speech"),),
        )
    ]

    def frames_for(clip, frame_ms):
        return [
            FrameObservation(
                timestamp_ms=index * frame_ms,
                channel_energies=[0.1],
                features={"feature": value},
                source="pcm_wav",
            )
            for index, value in enumerate((0.01, 0.012, 0.009, 0.011, 0.010))
        ]

    report = evaluate_online_predictor_safety(
        clips,
        frames_for,
        features=("feature",),
        maximum_mean_regret_fraction=1.0,
        maximum_clip_regret_fraction=1.0,
    )

    feature = report["feature_reports"]["feature"]
    assert report["all_feature_safety_passed"] is True
    assert feature["evaluated_clip_count"] == 1
    assert "superiority_passed" in feature
