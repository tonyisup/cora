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
    "operational_acceptance": "2026-07-12-operational-acceptance-attempt-2-result.json",
}

EXPECTED_EVIDENCE_SHA256 = {
    "stage1_prior_art": "d913f7e433e949a2afa88cd0e85c83f14b98b824e30d0a24993aeaa1d27ac6b2",
    "stage2_vocal_gate": "603cf12603f950515e1407bfac9b1a71c0137a2b6074087ecc7abb9888392068",
    "stage2_disposition": "87b57720404c1e9bbd8cbdc395ffc6762447a6103628782a6283533422b6d835",
    "stage3_dropout": "4f60aa7c9c44fe5ea8cdafaca705c12df8665659aeaa64671560eb0a68e39f9c",
    "stage3_wellington": "25f05865c9f984081a3fbdb8005577827f067f070430c836d2d0003c7240882e",
    "stage3_mentawai": "abaf369a8e8271344ffb34661aebe661f405955e8386edd87916607968e4c9c7",
    "stage4_exact_efference": "0525fa614ca1ab1655415448eb0c9c7c2e2d4c357bdd73f723ae036fe8d821c7",
    "stage5_plant": "991851cdf530e5cbbed5e345eadf9bb3d6f73df771b15f0078aa8a540bd5db88",
    "stage5_disposition": "ee0bc6f649d15b6b5195e471953255ce04818979674bb24c3378c4a4015cdfd4",
    "stage6_timing": "7f46030d3d159360cb3a5264124998f07db6a6fcf9760bd0b929ff1d469777f7",
    "stage6_energy": "12b6ac10068267589b1aaf484ad30c96cd6ee4678e8155dc85cca5b267633011",
    "operational_acceptance": "2fac84f4cc0555c870b33aa9cb8b7b92a3d8d57d373f3b147c762ac252fb0627",
}


def repository_evidence_dir() -> Path:
    """Return the evidence directory adjacent to this source checkout."""

    return Path(__file__).resolve().parents[2] / "docs" / "evidence"


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

    if set(EXPECTED_EVIDENCE_SHA256) != set(EVIDENCE_FILES):
        raise ValueError("operational evidence hash-pin roles are incomplete")

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
    acceptance = loaded["operational_acceptance"]

    timing_aggregate = timing.get("aggregate")
    if not isinstance(timing_aggregate, dict):
        raise ValueError("Stage 6 timing evidence is missing its aggregate")
    salient_range = timing_aggregate.get(
        "salient_adc_frame_end_to_workspace_p95_ms_range"
    )
    if not isinstance(salient_range, list) or not salient_range:
        raise ValueError("Stage 6 timing evidence is missing its salient p95 range")

    checks = {
        **{
            f"{role}_sha256_matches_pin": hashes[role] == expected
            for role, expected in EXPECTED_EVIDENCE_SHA256.items()
        },
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
        "current_operational_acceptance_passed": acceptance.get(
            "operational_acceptance_passed"
        )
        is True,
        "current_operational_acceptance_integrity_clean": acceptance.get(
            "criteria", {}
        ).get("all_required_integrity_counts_zero")
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
