"""Contracts for offline bounded vocal-imitation product-session reports."""

from __future__ import annotations

import pytest

from cora_infans.product_session import (
    build_product_session_report,
    validate_product_session_report,
)


_SHA256 = "a" * 64


def _artifacts() -> dict[str, dict[str, str]]:
    return {
        "source_human_wav": {"path": "private/source_human.wav", "sha256": _SHA256},
        "target_captured_wav": {"path": "private/target_captured.wav", "sha256": _SHA256},
        "agent_emitted_wav": {"path": "private/agent_emitted.wav", "sha256": _SHA256},
        "response_captured_wav": {"path": "private/response_captured.wav", "sha256": _SHA256},
        "self_prediction_wav": {"path": "private/self_prediction.wav", "sha256": _SHA256},
        # A report cannot self-attest its final SHA-256 without a circular digest.
        # A later sibling manifest may attest this completed report instead.
        "report_json": {"path": "report.json"},
    }


def _scores() -> dict[str, dict[str, float]]:
    return {
        "imitation_fidelity": {"mel_cepstral_distance": 0.24},
        "self_model_fidelity": {"mel_cepstral_distance": 0.11},
    }


def test_synthetic_fixture_report_has_required_artifacts_and_separate_scores() -> None:
    report = build_product_session_report(
        evidence_class="synthetic_fixture",
        artifacts=_artifacts(),
        scores=_scores(),
        association_frozen_before_evaluation=True,
    )

    assert report["evidence_class"] == "synthetic_fixture"
    assert report["v4_evidence_mutated"] is False
    assert set(report["artifacts"]) == set(_artifacts())
    assert set(report["scores"]) == {"imitation_fidelity", "self_model_fidelity"}
    assert report["association_frozen_before_evaluation"] is True
    assert validate_product_session_report(report) == report


def test_real_authorized_session_requires_authorization_and_operating_point() -> None:
    with pytest.raises(ValueError, match="authorization"):
        build_product_session_report(
            evidence_class="real_authorized_session",
            artifacts=_artifacts(),
            scores=_scores(),
            association_frozen_before_evaluation=True,
        )

    report = build_product_session_report(
        evidence_class="real_authorized_session",
        artifacts=_artifacts(),
        scores=_scores(),
        association_frozen_before_evaluation=True,
        authorization={"recording_authorized": True, "protocol_version": "product-session-v1"},
        operating_point={"input_device": "XVF3800 :0", "sample_rate_hz": 16000},
    )

    assert report["authorization"]["recording_authorized"] is True
    assert report["operating_point"]["sample_rate_hz"] == 16000


def test_report_rejects_missing_artifact_or_unfrozen_association() -> None:
    missing = _artifacts()
    missing.pop("self_prediction_wav")
    with pytest.raises(ValueError, match="required artifacts"):
        build_product_session_report(
            evidence_class="synthetic_fixture",
            artifacts=missing,
            scores=_scores(),
            association_frozen_before_evaluation=True,
        )

    with pytest.raises(ValueError, match="frozen"):
        build_product_session_report(
            evidence_class="synthetic_fixture",
            artifacts=_artifacts(),
            scores=_scores(),
            association_frozen_before_evaluation=False,
        )
