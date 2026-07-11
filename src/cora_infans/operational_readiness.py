"""Fail-loud readiness summary for the bounded operational specification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .operational_contract import OPERATIONAL_CONTRACT


EVIDENCE_FILES = {
    "stage1_prior_art": "2026-07-10-prior-art-replay.json",
    "stage2_vocal_gate": "2026-07-10-yin-vocal-gate-v2-result.json",
    "stage2_disposition": "2026-07-11-stage2-broad-register-disposition.json",
    "stage3_dropout": "2026-07-10-dropout-v5-real-result.json",
    "stage3_wellington": "2026-07-10-natural-event-cross-fieldsite-result.json",
    "stage3_mentawai": "2026-07-10-natural-event-men-replication-result.json",
    "stage4_exact_efference": "2026-07-11-natural-speech-workspace-spectral-v3-real-result.json",
    "stage5_plant": "2026-07-11-h4-option-f-expanded-band-plant-result.json",
    "stage5_disposition": "2026-07-11-h4-bounded-local-family-disposition.json",
    "stage6_timing": "2026-07-11-stage6-natural-speech-hardware-volume-diagnostic-result.json",
    "stage6_energy": "2026-07-11-stage6-energy-result.json",
}


def _load_evidence(root: Path, name: str) -> tuple[dict[str, object], str]:
    path = root / name
    if not path.is_file():
        raise ValueError(f"required operational evidence is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operational evidence JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"operational evidence must be an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def build_operational_readiness_report(
    evidence_dir: str | Path,
) -> dict[str, object]:
    """Validate retained evidence for every bounded operational stage."""

    root = Path(evidence_dir).resolve()
    loaded: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for role, filename in EVIDENCE_FILES.items():
        loaded[role], hashes[role] = _load_evidence(root, filename)

    prior = loaded["stage1_prior_art"]
    vocal = loaded["stage2_vocal_gate"]
    stage2 = loaded["stage2_disposition"]
    dropout = loaded["stage3_dropout"]
    wellington = loaded["stage3_wellington"]
    mentawai = loaded["stage3_mentawai"]
    exact = loaded["stage4_exact_efference"]
    plant = loaded["stage5_plant"]
    stage5 = loaded["stage5_disposition"]
    timing = loaded["stage6_timing"]
    energy = loaded["stage6_energy"]

    timing_aggregate = timing.get("aggregate")
    if not isinstance(timing_aggregate, dict):
        raise ValueError("Stage 6 timing evidence is missing its aggregate")
    salient_range = timing_aggregate.get(
        "salient_adc_frame_end_to_workspace_p95_ms_range"
    )
    if not isinstance(salient_range, list) or not salient_range:
        raise ValueError("Stage 6 timing evidence is missing its salient p95 range")

    checks = {
        "stage0_contract_valid": OPERATIONAL_CONTRACT["status"]
        == "bounded_operational_contract",
        "stage1_real_frames_retained": int(prior["source"]["clip_count"]) > 0,
        "stage2_vocal_front_gate_passed": vocal.get("verdict")
        == "all_preregistered_gates_passed",
        "stage2_register_program_stopped": bool(
            stage2["disposition"][
                "active_stage2_register_experimentation_stopped"
            ]
        ),
        "stage3_known_efference_dropout_passed": dropout.get(
            "all_acceptance_gates_passed"
        )
        is True,
        "stage3_wellington_transfer_passed": wellington["report"].get(
            "all_acceptance_gates_passed"
        )
        is True,
        "stage3_mentawai_transfer_passed": mentawai["report"].get(
            "all_acceptance_gates_passed"
        )
        is True,
        "stage4_exact_efference_passed": exact.get("verdict")
        == "real_io_acceptance_passed",
        "stage4_exact_replay_required": OPERATIONAL_CONTRACT["stage4"][
            "exact_commanded_utterance_replay_required"
        ]
        is True,
        "stage5_factorized_plant_passed": plant.get("all_plant_gates_passed")
        is True,
        "stage5_inverse_program_stopped": bool(
            stage5["disposition"]["active_h4_experimentation_stopped"]
        ),
        "stage6_retained_timing_gate_passed": timing.get("overall_criteria_passed")
        is True,
        "stage6_operational_p95_at_most_10_5_ms": max(
            float(value) for value in salient_range
        )
        <= float(OPERATIONAL_CONTRACT["stage6"]["latency_max_ms"]),
        "stage6_zero_clipped_samples": int(
            timing_aggregate["clipped_sample_count"]
        )
        == 0,
        "stage6_energy_measurement_passed": energy.get("overall_criteria_passed")
        is True,
        "protected_stage2_pristine_fieldsites_retained": stage2["protected_data"]
        == {
            "san_diego_usd_pristine": True,
            "toronto_tor_pristine": True,
            "wellington_protected_but_previously_spent_by_h2": True,
        },
        "protected_h4_data_retained": stage5.get(
            "protected_data_accessed_by_decision"
        )
        is False,
    }
    return {
        "mode": "bounded_operational_readiness",
        "contract_schema_version": OPERATIONAL_CONTRACT["schema_version"],
        "evidence_dir": str(root),
        "checks": checks,
        "operational_evidence_ready": all(checks.values()),
        "evidence_sha256": hashes,
        "stopped_programs": list(OPERATIONAL_CONTRACT["stopped_programs"]),
        "test_suite": "not_executed_by_readiness_report",
        "claim_boundary": (
            "Readiness is limited to retained evidence and the named operating "
            "point; it does not extend unsupported claims or replace the test suite"
        ),
    }
