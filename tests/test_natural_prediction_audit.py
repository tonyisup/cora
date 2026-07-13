from __future__ import annotations

import numpy as np

from cora_infans.cli import natural_prediction_design_cli
from cora_infans.io.motor_plan import write_wav
from cora_infans.natural_prediction_audit import (
    NaturalPredictionCandidate,
    audit_natural_prediction_grid,
    evaluate_natural_prediction_candidate,
)


def _write_participant_clips(root) -> tuple[str, ...]:
    sample_rate = 16000
    participants = ("P01", "P02", "P03", "P04")
    frequencies = (300.0, 500.0, 800.0, 1100.0)
    token_samples = sample_rate // 10
    for participant_index, participant in enumerate(participants):
        for condition_index, suffix in enumerate(("B", "D")):
            sequence = []
            state = 0
            for index in range(40):
                t = np.arange(token_samples, dtype=np.float64) / sample_rate
                frequency = frequencies[state] + participant_index * 3.0
                sequence.append(
                    np.sin(2.0 * np.pi * frequency * t) * (0.15 + condition_index * 0.01)
                )
                state = (state + 1) % 4 if index % 5 else (state + 2) % 4
            write_wav(
                str(root / f"{participant}{suffix}.wav"),
                np.concatenate(sequence).astype(np.float32),
                sample_rate,
            )
    return participants


def test_natural_candidate_is_label_blind_and_participant_disjoint(tmp_path) -> None:
    participants = _write_participant_clips(tmp_path)
    report = evaluate_natural_prediction_candidate(
        root=tmp_path,
        participants=participants,
        candidate=NaturalPredictionCandidate("full_spectrum", 100, 1),
    )

    assert report["labels_consumed_by_learner"] is False
    assert report["participant_count"] == 4
    assert report["clip_count"] == 8
    assert len(report["folds"]) == 4
    for fold in report["folds"]:
        assert fold["held_participant"] not in fold["development_participants"]
        assert fold["participant_overlap_count"] == 0
        assert len(fold["clips"]) == 2


def test_natural_grid_is_deterministic_and_never_promotes_spent_data(tmp_path) -> None:
    participants = _write_participant_clips(tmp_path)
    kwargs = {
        "root": tmp_path,
        "participants": participants,
        "frame_ms_values": (100,),
        "horizon_values": (1,),
        "signature_modes": ("full_spectrum",),
    }

    first = audit_natural_prediction_grid(**kwargs)
    second = audit_natural_prediction_grid(**kwargs)

    assert first == second
    assert first["candidate_count"] == 1
    assert first["promotion_allowed"] is False
    assert first["evidence_class"] == "spent_development_real_audio"


def test_natural_design_cli_requires_explicit_root_and_participants(
    tmp_path, monkeypatch
) -> None:
    observed = {}

    def fake_audit(**kwargs):
        observed.update(kwargs)
        return {"verdict": "test_double"}

    monkeypatch.setattr("cora_infans.cli.audit_natural_prediction_grid", fake_audit)
    report = natural_prediction_design_cli(
        [
            "--evaluate-natural-prediction-design",
            "--root",
            str(tmp_path),
            "--participants",
            "P01,P02,P03",
        ]
    )

    assert report["verdict"] == "test_double"
    assert observed == {
        "root": str(tmp_path),
        "participants": ("P01", "P02", "P03"),
    }
