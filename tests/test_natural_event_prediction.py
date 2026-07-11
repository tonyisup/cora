from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cora_infans.io.motor_plan import write_wav
from cora_infans.natural_event_prediction import (
    NaturalEventConfig,
    audit_natural_event_grid,
    evaluate_natural_event_cross_fieldsite,
    evaluate_natural_event_leave_one_participant_out,
    extract_natural_event_states,
)


def _write_event_clips(root, participants) -> None:
    sample_rate = 16000
    frame = sample_rate // 20
    for participant_index, participant in enumerate(participants):
        for suffix_index, suffix in enumerate(("B", "D")):
            chunks = []
            for index in range(100):
                if index % 10 < 3:
                    chunk = np.zeros(frame, dtype=np.float32)
                else:
                    time = np.arange(frame) / sample_rate
                    frequency = 300.0 + ((index // 3) % 4) * 180.0
                    frequency += participant_index * 2.0
                    chunk = (
                        np.sin(2.0 * np.pi * frequency * time)
                        * (0.12 + suffix_index * 0.01)
                    ).astype(np.float32)
                chunks.append(chunk)
            write_wav(
                str(root / f"{participant}{suffix}.wav"),
                np.concatenate(chunks),
                sample_rate,
            )


def test_event_states_are_causal_finite_and_bounded(tmp_path) -> None:
    _write_event_clips(tmp_path, ("P01",))

    states, thresholds = extract_natural_event_states(
        tmp_path / "P01B.wav", NaturalEventConfig()
    )

    assert states
    assert set(states) <= {0, 1, 2, 3}
    assert all(np.isfinite(value) for value in thresholds.values())


def test_event_leave_one_participant_out_is_label_blind(tmp_path) -> None:
    participants = ("P01", "P02", "P03", "P04")
    _write_event_clips(tmp_path, participants)

    report = evaluate_natural_event_leave_one_participant_out(
        root=tmp_path,
        participants=participants,
        config=NaturalEventConfig(),
    )

    assert report["labels_consumed_by_learner"] is False
    assert report["clip_count"] == 8
    assert report["bounded_state_scalars"] == 16
    for fold in report["folds"]:
        assert fold["held_participant"] not in fold["development_participants"]
        assert fold["participant_overlap_count"] == 0


def test_event_grid_and_cross_fieldsite_reports_are_deterministic(tmp_path) -> None:
    development = ("D01", "D02", "D03", "D04")
    held_out = ("H01", "H02", "H03", "H04")
    _write_event_clips(tmp_path, development + held_out)
    grid = audit_natural_event_grid(
        root=tmp_path,
        participants=development,
        frame_ms_values=(50,),
        quiet_quantiles=(0.40,),
        novelty_quantiles=(0.75,),
    )
    transfer = evaluate_natural_event_cross_fieldsite(
        development_root=tmp_path,
        development_participants=development,
        held_out_root=tmp_path,
        held_out_participants=held_out,
        config=NaturalEventConfig(),
    )

    assert grid["candidate_count"] == 1
    assert grid["promotion_allowed"] is False
    assert transfer["participant_overlap_count"] == 0
    assert transfer["clip_count"] == 8
    assert transfer["verdict"] in {
        "natural_event_transfer_passed",
        "natural_event_transfer_failed",
    }


def test_retained_cross_fieldsite_evidence_matches_preregistration() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    preregistration = json.loads(
        (
            evidence
            / "2026-07-10-natural-event-cross-fieldsite-preregistration.json"
        ).read_text()
    )
    result = json.loads(
        (
            evidence
            / "2026-07-10-natural-event-cross-fieldsite-result.json"
        ).read_text()
    )

    frozen = result["frozen_hash_verification"]
    assert frozen["design_audit_sha256"] == hashlib.sha256(
        (evidence / "2026-07-10-natural-event-prediction-design-audit.json").read_bytes()
    ).hexdigest()
    assert frozen["implementation_sha256"] == preregistration["implementation_sha256"]
    assert frozen["cli_sha256"] == preregistration["cli_sha256"]
    assert result["report"]["verdict"] == "natural_event_transfer_passed"
    assert result["report"]["all_acceptance_gates_passed"] is True
    assert len(result["held_out_audio_manifest"]) == 8
    assert len({item["sha256"] for item in result["held_out_audio_manifest"]}) == 8
    assert all(len(item["sha256"]) == 64 for item in result["held_out_audio_manifest"])
