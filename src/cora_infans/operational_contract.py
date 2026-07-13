"""Machine-enforced boundaries from the bounded operational VISION."""

from __future__ import annotations

import json
from importlib.resources import files
import math
import re
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
    operating_point = report.get("operating_point")
    if not isinstance(operating_point, Mapping):
        raise ValueError("missing Stage 6 physical operating-point metadata")
    metadata_criteria = _operating_point_criteria(operating_point)
    metric = str(stage6["latency_metric"])
    try:
        latency_value = float(report[metric])
    except (KeyError, TypeError, ValueError):
        latency_value = float("nan")
    latency_available = math.isfinite(latency_value) and latency_value >= 0.0
    latency = latency_value if latency_available else None

    zero_fields = tuple(stage6["required_zero_counts"])
    missing = [field for field in zero_fields if field not in report]
    if missing:
        raise ValueError("missing Stage 6 integrity fields: " + ", ".join(missing))
    profile = stage6["acceptance_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("Stage 6 contract is missing its acceptance profile")
    criteria = {
        **metadata_criteria,
        "real_capture": report.get("real_capture") is True,
        "callback_backend_matches": report.get("capture_backend")
        == stage6["capture_backend"],
        "frame_ms_matches": report.get("frame_ms") == profile["frame_ms"],
        "frame_count_matches": report.get("frame_count")
        == profile["target_frame_count"],
        "minimum_salient_frames_met": int(report.get("salient_frame_count", 0))
        >= int(profile["minimum_salient_frame_count"]),
        "active_playback_frames_present": int(
            report.get("active_playback_frame_count", 0)
        )
        > 0,
        "software_playback_gain_matches": float(
            report.get("software_playback_gain_db", float("nan"))
        )
        == float(profile["software_playback_gain_db"]),
        "stimulus_hash_matches": report.get("stimulus_sha256")
        == profile["stimulus_sha256"],
        "salient_latency_available": latency_available,
        "salient_p95_at_most_10_5_ms": latency_available
        and latency_value <= float(stage6["latency_max_ms"]),
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


def _operating_point_criteria(
    operating_point: Mapping[str, object],
) -> dict[str, bool]:
    """Compare a self-described run with every physical VISION invariant."""

    stage6 = OPERATIONAL_CONTRACT["stage6"]
    attestations = operating_point.get("operator_attestations")
    if not isinstance(attestations, Mapping):
        raise ValueError("operating point is missing operator attestations")
    return {
        "operating_point_schema_matches": operating_point.get("schema_version")
        == stage6["operating_point_schema_version"],
        "host_matches": operating_point.get("host") == stage6["host"],
        "input_matches": operating_point.get("input") == stage6["input"],
        "self_source_matches": operating_point.get("self_source")
        == stage6["self_source"],
        "world_source_matches": operating_point.get("world_source")
        == stage6["world_source"],
        "self_geometry_operator_confirmed": attestations.get(
            "self_source_geometry_confirmed"
        )
        is True,
        "self_volume_operator_confirmed": attestations.get(
            "self_source_system_volume_confirmed"
        )
        is True,
        "world_geometry_operator_confirmed": attestations.get(
            "world_source_geometry_confirmed"
        )
        is True,
        "world_volume_operator_confirmed": attestations.get(
            "world_source_hardware_volume_at_approximately_50_percent"
        )
        is True,
    }


def evaluate_stage4_exact_efference(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Validate an exact-replay self-voice report without broadening H3."""

    stage4 = OPERATIONAL_CONTRACT["stage4"]
    operating_point = report.get("operating_point")
    if not isinstance(operating_point, Mapping):
        raise ValueError("missing Stage 4 physical operating-point metadata")
    replay = report.get("exact_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("missing Stage 4 exact-replay metadata")
    commanded_hash = str(replay.get("commanded_utterance_sha256", ""))
    replay_hash = str(replay.get("prior_replay_utterance_sha256", ""))
    sensory_hash = str(replay.get("sensory_consequence_sha256", ""))
    hash_pattern = re.compile(r"[0-9a-f]{64}")
    capacity = int(report.get("workspace_capacity", 0))
    criteria = {
        **_operating_point_criteria(operating_point),
        "exact_replay_performed": replay.get("performed") is True,
        "commanded_utterance_hash_valid": hash_pattern.fullmatch(commanded_hash)
        is not None,
        "prior_replay_hash_valid": hash_pattern.fullmatch(replay_hash) is not None,
        "prior_replay_matches_command": replay_hash == commanded_hash,
        "efference_identity_present": bool(replay.get("efference_id")),
        "sensory_consequence_hash_valid": hash_pattern.fullmatch(sensory_hash)
        is not None,
        "workspace_capacity_within_limit": 0 < capacity
        <= int(stage4["workspace_capacity"]),
        **{
            f"{field}_is_zero": int(report.get(field, -1)) == 0
            for field in stage4["required_zero_counts"]
        },
    }
    return {
        "criteria": criteria,
        "exact_efference_passed": all(criteria.values()),
        "claim_boundary": (
            "Exact commanded-utterance replay only; no no-replay, semantic, "
            "speaker-generalization, or geometry-transfer claim"
        ),
    }
