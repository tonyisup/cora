"""Contracts for the bounded offline speech-learning demonstration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.speech_learning_demo import (
    DEMO_VOWELS,
    REPERTOIRE_SIZE,
    LocalImitationLearner,
    MotorRepertoire,
    build_demo_target,
    run_speech_learning_demo,
)


def test_babbling_repertoire_is_bounded_deterministic_forward_memory() -> None:
    first = MotorRepertoire.babble()
    second = MotorRepertoire.babble()

    assert len(first.prototypes) == REPERTOIRE_SIZE == 96
    assert first.digest() == second.digest()
    assert [item.halton_index for item in first.prototypes] == [
        14001 + 37 * offset for offset in range(96)
    ]
    assert len({item.motor for item in first.prototypes}) == 96
    assert all(len(item.sensory) == 30 for item in first.prototypes)
    assert all(
        np.linalg.norm(item.sensory_array()) == pytest.approx(1.0)
        for item in first.prototypes
    )


def test_local_learner_receives_no_label_and_recalls_bounded_association() -> None:
    repertoire = MotorRepertoire.babble()
    learner = LocalImitationLearner(repertoire)
    target = build_demo_target(
        repertoire, vowel="a", scenario="cross-configuration"
    )

    result = learner.imitate(
        target.sensory_array(), target_id=target.target_id
    )

    assert result["learner_input"]["vowel_label_provided"] is False
    assert len(result["attempts"]) == 12
    assert [row["phase"] for row in result["attempts"][-3:]] == [
        "learned_recall",
        "learned_recall",
        "learned_recall",
    ]
    assert len({row["prototype_id"] for row in result["attempts"][-3:]}) == 1
    assert result["improvement_ratio"] >= 0.30
    assert result["final_best_sensory_distance"] <= 0.20
    assert result["session_passed"] is True


def test_both_demo_scenarios_pass_all_three_targets(tmp_path) -> None:
    for scenario in ("synthetic-mirror", "cross-configuration"):
        report = run_speech_learning_demo(
            scenario=scenario,
            output_dir=tmp_path / scenario,
        )

        assert report["demo_passed"] is True
        assert report["vowels"] == list(DEMO_VOWELS)
        assert report["association_count"] == 3
        assert all(session["session_passed"] for session in report["sessions"])
        assert all(
            session["label_role"] == "display_and_evaluation_only"
            for session in report["sessions"]
        )
        assert report["aggregate_criteria"]["recorded_audio_not_accessed"] is True
        assert report["aggregate_criteria"]["protected_data_not_accessed"] is True
        assert report["aggregate_criteria"]["live_io_not_executed"] is True


def test_demo_artifacts_are_hash_identified_and_reproducible(tmp_path) -> None:
    first = run_speech_learning_demo(
        scenario="cross-configuration", output_dir=tmp_path / "first"
    )
    second = run_speech_learning_demo(
        scenario="cross-configuration", output_dir=tmp_path / "second"
    )

    assert first == second
    for session in first["sessions"]:
        for role in ("target", "learned"):
            filename = (
                session["target"]["waveform"]["filename"]
                if role == "target"
                else session["learned_output"]["waveform"]["filename"]
            )
            expected = (
                session["target"]["waveform"]["sha256"]
                if role == "target"
                else session["learned_output"]["waveform"]["sha256"]
            )
            digest = hashlib.sha256(
                (tmp_path / "first" / filename).read_bytes()
            ).hexdigest()
            assert digest == expected
            assert (tmp_path / "first" / filename).read_bytes() == (
                tmp_path / "second" / filename
            ).read_bytes()

    retained = json.loads(
        (tmp_path / "first/speech-learning-session.json").read_text()
    )
    assert retained == first
    dashboard = (tmp_path / "first/speech-learning-demo.html").read_text()
    assert "Cora speech-learning demo" in dashboard
    assert "a-target.wav" in dashboard
    assert "a-learned.wav" in dashboard
    assert (tmp_path / "first/speech-learning-demo.html").read_bytes() == (
        tmp_path / "second/speech-learning-demo.html"
    ).read_bytes()


def test_prior_demo_candidates_remain_stopped() -> None:
    root = Path(__file__).resolve().parents[1]
    for version in ("v1", "v2"):
        result = json.loads(
            (
                root
                / f"docs/evidence/2026-07-12-speech-learning-demo-{version}-design-failure.json"
            ).read_text()
        )
        assert result["status"] == "candidate_stopped"
        assert result["protected_data_accessed"] is False
        assert result["live_io_executed"] is False


def test_retained_v3_result_preserves_demo_claim_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-12-speech-learning-demo-v3-result.json"
        ).read_text()
    )

    assert result["demo_passed"] is True
    assert result["synthetic_mirror"]["demo_passed"] is True
    assert result["cross_configuration"]["demo_passed"] is True
    assert result["criteria"]["learner_received_no_vowel_labels"] is True
    assert result["criteria"]["recorded_audio_not_accessed"] is True
    assert "not continuous inverse closure" in result["claim_boundary"]
