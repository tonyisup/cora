from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cora_infans.nonlinear_predictive_codebook import (
    CONTEXT_FRAMES,
    PROTOTYPE_COUNT,
    CompetitiveTemporalCodebook,
    NonlinearPredictiveCodebookModel,
    causal_context_targets,
    nonlinear_codebook_leave_one_fieldsite_out,
    select_nonlinear_codebook_thresholds,
)
from cora_infans.predictive_register_dynamics import (
    CONDITION_LABELS,
    ProsodicSequenceExample,
)


def test_causal_contexts_never_include_prediction_target() -> None:
    states = np.arange(50, dtype=np.float64).reshape(10, 5)
    contexts, targets = causal_context_targets(states)

    assert contexts.shape == (6, 20)
    assert np.array_equal(contexts[0], states[:CONTEXT_FRAMES].reshape(-1))
    assert np.array_equal(targets[0], states[CONTEXT_FRAMES])
    assert not np.any(contexts[0] == targets[0, 0])


def test_competitive_codebook_is_deterministic_bounded_and_locally_updated() -> None:
    rng = np.random.default_rng(4)
    contexts = np.concatenate(
        [rng.normal(loc=value, scale=0.2, size=(80, 20)) for value in (-2, 0, 2)]
    )
    first = CompetitiveTemporalCodebook.fit(contexts)
    second = CompetitiveTemporalCodebook.fit(contexts)

    assert first == second
    assert len(first.centers) == PROTOTYPE_COUNT
    assert all(len(center) == 20 for center in first.centers)
    assert sum(first.winner_update_count) == len(contexts)
    assert first.quantization_error_after <= first.quantization_error_before
    assert first.utilization_fraction >= 0.75


def test_nonlinear_activation_retains_history_when_last_state_matches() -> None:
    rng = np.random.default_rng(8)
    contexts = rng.normal(size=(100, 20))
    codebook = CompetitiveTemporalCodebook.fit(contexts)
    pair = contexts[:2].copy()
    pair[1, -5:] = pair[0, -5:]

    activations = codebook.activations(pair)

    assert np.array_equal(pair[0, -5:], pair[1, -5:])
    assert not np.allclose(activations[0], activations[1])


def _site(fieldsite: str, offset: float) -> list[ProsodicSequenceExample]:
    examples = []
    for speaker_index in range(3):
        for condition_index, label in enumerate(CONDITION_LABELS):
            time = np.linspace(0.0, 1.0, 32)
            infant = 1.0 if label.startswith("infant") else -1.0
            mode = 1.0 if label.endswith("speech") else -1.0
            states = np.column_stack(
                [
                    offset
                    + speaker_index * 0.1
                    + mode * np.sin((feature + 1) * np.pi * time)
                    + infant * time ** (1 + feature % 2)
                    for feature in range(5)
                ]
            )
            examples.append(
                ProsodicSequenceExample(
                    clip_id=f"{fieldsite}{speaker_index:02d}{'ABCD'[condition_index]}",
                    speaker_id=f"{fieldsite}{speaker_index:02d}",
                    fieldsite=fieldsite,
                    label=label,
                    states=states,
                    vocal_score=0.5,
                )
            )
    return examples


def test_target_ids_ads_labels_do_not_change_nonlinear_scores() -> None:
    training = [*_site("AAA", 0.0), *_site("BBB", 3.0)]
    target = _site("CCC", 6.0)
    relabeled = [
        ProsodicSequenceExample(
            clip_id=item.clip_id,
            speaker_id=item.speaker_id,
            fieldsite=item.fieldsite,
            label=(
                item.label.replace("infant_directed", "adult_directed")
                if item.label.startswith("infant")
                else item.label.replace("adult_directed", "infant_directed")
            ),
            states=item.states,
            vocal_score=item.vocal_score,
        )
        for item in target
    ]
    model = NonlinearPredictiveCodebookModel.fit(training)
    original = model.evaluate_batch(target)
    swapped = model.evaluate_batch(relabeled)

    original_scores = {
        item["clip_id"]: (item["speech_score"], item["ids_score"])
        for item in original["predictions"]
    }
    swapped_scores = {
        item["clip_id"]: (item["speech_score"], item["ids_score"])
        for item in swapped["predictions"]
    }
    assert original_scores == swapped_scores


def test_nonlinear_codebook_uses_grouped_folds_and_frozen_thresholds() -> None:
    examples = [*_site("AAA", 0.0), *_site("BBB", 3.0), *_site("CCC", 6.0)]
    report = nonlinear_codebook_leave_one_fieldsite_out(examples)
    selection = select_nonlinear_codebook_thresholds(report["predictions"])

    assert report["fieldsite_count"] == 3
    assert report["prediction_count"] == 36
    assert report["representation_uses_condition_labels"] is False
    assert report["confirmation_audio_or_metadata_accessed"] is False
    assert report["promotion_allowed"] is False
    assert selection["protocol"] == "nonlinear_codebook_out_of_field_threshold_selection"
    assert selection["metrics"]["false_positive_rate"] <= 0.15


def test_retained_nonlinear_result_preserves_representation_and_stop_decisions() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-nonlinear-predictive-codebook-result.json"
        ).read_text()
    )

    assert result["source"]["clip_count"] == 200
    assert result["codebook_health"]["all_health_gates_passed"] is True
    assert result["out_of_field_threshold_selection"]["f1"] == 0.5346534653465347
    assert result["out_of_field_threshold_selection"]["minimum_fieldsite_f1"] == 0.0
    assert result["development_candidate_ready"] is False
    assert result["confirmation_audio_or_metadata_accessed"] is False
    assert result["promotion_allowed"] is False
