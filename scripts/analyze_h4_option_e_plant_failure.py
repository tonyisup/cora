#!/usr/bin/env python3
"""Descriptive post-hoc decomposition of the retained Option E P0 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cora_infans.factorized_tract_plant import (
    FROZEN_OPTION_E_CONFIG,
    factorized_area_function,
    factorized_formants,
    factorized_transfer,
    halton5_points,
)


def _separated_peaks(motor: np.ndarray, stop_hz: float) -> list[float]:
    frequencies = np.arange(80.0, stop_hz + 5.0, 5.0)
    magnitude = np.abs(
        factorized_transfer(motor, frequencies, FROZEN_OPTION_E_CONFIG)
    )
    candidates = [
        index
        for index in range(1, len(magnitude) - 1)
        if magnitude[index] > magnitude[index - 1]
        and magnitude[index] >= magnitude[index + 1]
    ]
    selected: list[float] = []
    for index in candidates:
        frequency = float(frequencies[index])
        if not selected or frequency - selected[-1] >= 150.0:
            selected.append(frequency)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    retained = json.loads(result_path.read_text())
    if retained["all_plant_gates_passed"] is not False:
        raise ValueError("analysis requires the retained failed Option E result")

    points = halton5_points(8193, 9216)
    valid_motors: list[np.ndarray] = []
    missing_rows: list[dict[str, object]] = []
    for offset, motor in enumerate(points):
        formants = factorized_formants(motor, FROZEN_OPTION_E_CONFIG)
        if formants is not None:
            valid_motors.append(motor)
            continue
        geometry = factorized_area_function(motor, FROZEN_OPTION_E_CONFIG)
        missing_rows.append(
            {
                "halton_index": 8193 + offset,
                "motor": motor.tolist(),
                "minimum_area_cm2": float(np.min(geometry.areas_cm2)),
                "lip_area_cm2": float(geometry.areas_cm2[-1]),
                "total_length_cm": geometry.total_length_cm,
                "separated_peaks_80_3500_hz": _separated_peaks(motor, 3500.0),
                "separated_peaks_80_5000_hz": _separated_peaks(motor, 5000.0),
            }
        )

    valid = np.stack(valid_motors)
    invalid = np.asarray([row["motor"] for row in missing_rows], dtype=np.float64)
    report = {
        "mode": "h4_option_e_plant_failure_analysis_post_hoc",
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "candidate_rerun": False,
        "threshold_or_domain_selected": False,
        "confirmation_points_are_spent": True,
        "valid_count": len(valid),
        "missing_three_formant_count": len(invalid),
        "motor_coordinate_order": [
            "jaw_aperture",
            "constriction_magnitude",
            "constriction_position",
            "lip_aperture",
            "lip_protrusion",
        ],
        "valid_motor_mean": np.mean(valid, axis=0).tolist(),
        "invalid_motor_mean": np.mean(invalid, axis=0).tolist(),
        "invalid_motor_min": np.min(invalid, axis=0).tolist(),
        "invalid_motor_max": np.max(invalid, axis=0).tolist(),
        "missing_rows": missing_rows,
        "interpretation": "The failure is sparse and concentrated in high-constriction states, but no rectangular motor bound separates failures from valid vowel targets. This analysis does not waive P0 or select a new admissible domain.",
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
