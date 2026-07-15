"""Offline contract validation for bounded vocal-imitation product sessions.

This module validates JSON-safe report metadata only. It does not capture audio,
select prototypes, score waveforms, or modify retained V4 evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


PRODUCT_SESSION_REPORT_VERSION = 1
EVIDENCE_CLASSES = frozenset(
    {
        "retained_v4_evidence",
        "synthetic_fixture",
        "real_authorized_session",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "source_human_wav",
        "target_captured_wav",
        "agent_emitted_wav",
        "response_captured_wav",
        "self_prediction_wav",
        "report_json",
    }
)
REQUIRED_SCORE_FAMILIES = frozenset({"imitation_fidelity", "self_model_fidelity"})


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_artifacts(artifacts: object) -> None:
    records = _require_mapping(artifacts, name="artifacts")
    missing = REQUIRED_ARTIFACTS.difference(records)
    if missing:
        raise ValueError(f"report is missing required artifacts: {sorted(missing)}")
    for name in REQUIRED_ARTIFACTS:
        record = _require_mapping(records[name], name=f"artifact {name}")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"artifact {name} requires a non-empty path")
        if name == "report_json":
            # The report cannot contain its own final digest without circular
            # content. A future sibling manifest attests the completed report.
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"artifact {name} requires a lowercase SHA-256 digest")


def _validate_scores(scores: object) -> None:
    families = _require_mapping(scores, name="scores")
    missing = REQUIRED_SCORE_FAMILIES.difference(families)
    if missing:
        raise ValueError(f"report is missing required score families: {sorted(missing)}")
    for family in REQUIRED_SCORE_FAMILIES:
        metrics = _require_mapping(families[family], name=f"score family {family}")
        if not metrics:
            raise ValueError(f"score family {family} must contain at least one metric")
        for metric, value in metrics.items():
            if not isinstance(metric, str) or not metric:
                raise ValueError(f"score family {family} has an invalid metric name")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"score metric {family}.{metric} must be finite")


def _validate_real_authorization(report: Mapping[str, object]) -> None:
    authorization = _require_mapping(report.get("authorization"), name="authorization")
    if authorization.get("recording_authorized") is not True:
        raise ValueError("real_authorized_session requires recording authorization")
    protocol_version = authorization.get("protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("real_authorized_session requires a protocol version")
    operating_point = _require_mapping(report.get("operating_point"), name="operating_point")
    input_device = operating_point.get("input_device")
    sample_rate = operating_point.get("sample_rate_hz")
    if not isinstance(input_device, str) or not input_device:
        raise ValueError("real_authorized_session requires an input device")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("real_authorized_session requires a positive sample rate")


def validate_product_session_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate and return a copy of an offline product-session report."""

    payload = _require_mapping(report, name="product-session report")
    evidence_class = payload.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unknown product-session evidence class: {evidence_class!r}")
    if payload.get("v4_evidence_mutated") is not False:
        raise ValueError("product-session reports must declare V4 evidence unmodified")
    if payload.get("association_frozen_before_evaluation") is not True:
        raise ValueError("association must be frozen before evaluation")
    _validate_artifacts(payload.get("artifacts"))
    _validate_scores(payload.get("scores"))
    if evidence_class == "real_authorized_session":
        _validate_real_authorization(payload)
    return dict(payload)


def build_product_session_report(
    *,
    evidence_class: str,
    artifacts: Mapping[str, object],
    scores: Mapping[str, object],
    association_frozen_before_evaluation: bool,
    authorization: Mapping[str, object] | None = None,
    operating_point: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a deterministic metadata report and enforce its contract."""

    report: dict[str, object] = {
        "report_version": PRODUCT_SESSION_REPORT_VERSION,
        "evidence_class": evidence_class,
        "v4_evidence_mutated": False,
        "association_frozen_before_evaluation": association_frozen_before_evaluation,
        "artifacts": dict(artifacts),
        "scores": dict(scores),
    }
    if authorization is not None:
        report["authorization"] = dict(authorization)
    if operating_point is not None:
        report["operating_point"] = dict(operating_point)
    return validate_product_session_report(report)
