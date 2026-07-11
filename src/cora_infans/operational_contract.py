"""Machine-enforced boundaries from the bounded operational VISION."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Mapping


def load_operational_contract() -> dict[str, object]:
    """Load and validate the packaged operational contract."""

    resource = files("cora_infans").joinpath("operational_contract.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    validate_operational_contract(contract)
    return contract


def validate_operational_contract(contract: Mapping[str, object]) -> None:
    """Fail loudly if critical governance invariants drift."""

    if contract.get("schema_version") != 1:
        raise ValueError("unsupported operational-contract schema")
    if contract.get("authority") != "VISION.md":
        raise ValueError("operational contract must be governed by VISION.md")

    claims = contract.get("claims")
    stage4 = contract.get("stage4")
    stage5 = contract.get("stage5")
    stage6 = contract.get("stage6")
    protected = contract.get("protected_data")
    governance = contract.get("governance")
    if not all(
        isinstance(value, Mapping)
        for value in (claims, stage4, stage5, stage6, protected, governance)
    ):
        raise ValueError("operational contract is missing required sections")

    assert isinstance(claims, Mapping)
    assert isinstance(stage4, Mapping)
    assert isinstance(stage5, Mapping)
    assert isinstance(stage6, Mapping)
    assert isinstance(protected, Mapping)
    assert isinstance(governance, Mapping)
    if claims.get("stage2_register_detection") != "falsified_within_tested_scope":
        raise ValueError("Stage 2 falsification boundary drifted")
    if stage4.get("exact_commanded_utterance_replay_required") is not True:
        raise ValueError("exact-efference replay must remain mandatory")
    if stage4.get("no_replay_authorized") is not False:
        raise ValueError("no-replay execution is not authorized")
    if stage5.get("local_inverse_execution_authorized") is not False:
        raise ValueError("local H4 inverse execution is stopped")
    if stage6.get("latency_metric") != (
        "salient_adc_frame_end_to_workspace_p95_ms"
    ):
        raise ValueError("Stage 6 latency metric drifted")
    if float(stage6.get("latency_max_ms", float("nan"))) != 10.5:
        raise ValueError("Stage 6 latency ceiling drifted")
    if set(protected.get("stage2_pristine_fieldsites", ())) != {"USD", "TOR"}:
        raise ValueError("Stage 2 pristine-fieldsite boundary drifted")
    if governance.get("post_hoc_threshold_selection_authorized") is not False:
        raise ValueError("post-hoc threshold selection must remain prohibited")


OPERATIONAL_CONTRACT = load_operational_contract()
STAGE2_PROTECTED_FIELDSITES = frozenset(
    OPERATIONAL_CONTRACT["protected_data"]["stage2_fieldsites"]
)


def assert_program_active(program: str) -> None:
    """Reject execution of a program stopped by final governance."""

    stopped = set(OPERATIONAL_CONTRACT["stopped_programs"])
    if program in stopped:
        raise ValueError(f"program is stopped by VISION.md: {program}")


def assert_stage2_fieldsite_allowed(fieldsite: str) -> None:
    """Reject protected Stage 2 fieldsites before acquisition or inspection."""

    code = fieldsite.strip().upper()
    if code in STAGE2_PROTECTED_FIELDSITES:
        raise ValueError(f"Stage 2 fieldsite is protected: {code}")


def assert_h4_speaker_allowed(corpus: str, speaker: str) -> None:
    """Reject protected VocalSet and CMU Arctic speakers."""

    protected = OPERATIONAL_CONTRACT["protected_data"]
    corpus_key = corpus.strip().lower()
    speaker_key = speaker.strip().lower()
    if corpus_key == "vocalset":
        blocked = set(protected["vocalset_speakers"])
    elif corpus_key in {"cmu", "cmu_arctic", "cmu-arctic"}:
        blocked = set(protected["cmu_arctic_confirmation_speakers"]) | set(
            protected["cmu_arctic_reserve_speakers"]
        )
    else:
        raise ValueError(f"unknown protected H4 corpus: {corpus}")
    if speaker_key in blocked:
        raise ValueError(f"protected {corpus} speaker: {speaker_key}")


def evaluate_stage6_operating_point(report: Mapping[str, object]) -> dict[str, object]:
    """Evaluate a future run against the bounded Stage 6 operating point."""

    stage6 = OPERATIONAL_CONTRACT["stage6"]
    metric = str(stage6["latency_metric"])
    try:
        latency = float(report[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing finite Stage 6 latency metric: {metric}") from exc
    if not (latency >= 0.0):
        raise ValueError(f"invalid Stage 6 latency metric: {latency}")

    zero_fields = tuple(stage6["required_zero_counts"])
    missing = [field for field in zero_fields if field not in report]
    if missing:
        raise ValueError("missing Stage 6 integrity fields: " + ", ".join(missing))
    criteria = {
        "salient_p95_at_most_10_5_ms": latency <= float(stage6["latency_max_ms"]),
        **{
            f"{field}_is_zero": int(report[field]) == 0 for field in zero_fields
        },
    }
    return {
        "metric": metric,
        "latency_ms": latency,
        "criteria": criteria,
        "operating_point_passed": all(criteria.values()),
        "claim_boundary": (
            "ADC frame end to workspace decision on the named callback path; "
            "frame accumulation and XMOS-internal latency excluded"
        ),
    }
