#!/usr/bin/env python3
"""Post-hoc structural analysis of the retained Option G failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cora_infans.expanded_local_closure import fit_expanded_forward_map
from cora_infans.factorized_tract_plant import expanded_sensory, halton5_points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    retained = json.loads(result_path.read_text())
    if retained["all_synthetic_gates_passed"] is not False:
        raise ValueError("analysis requires the retained failed Option G result")

    model, _ = fit_expanded_forward_map()
    points = halton5_points(20481, 20992)
    errors: list[float] = []
    nearest_distances: list[float] = []
    boundary_margins: list[float] = []
    constrictions: list[float] = []
    for motor in points:
        distances = np.sqrt(np.sum((model.centers - motor) ** 2, axis=1))
        ordered = np.sort(distances)
        errors.append(
            float(np.linalg.norm(expanded_sensory(motor) - model.predict(motor)))
        )
        nearest_distances.append(float(ordered[0]))
        boundary_margins.append(float((ordered[1] - ordered[0]) / ordered[1]))
        constrictions.append(float(motor[1]))

    error = np.asarray(errors)
    margin = np.asarray(boundary_margins)
    quartiles = np.quantile(margin, [0.0, 0.25, 0.5, 0.75, 1.0])
    margin_rows = []
    for index in range(4):
        upper_inclusive = index == 3
        mask = (margin >= quartiles[index]) & (
            margin <= quartiles[index + 1]
            if upper_inclusive
            else margin < quartiles[index + 1]
        )
        margin_rows.append(
            {
                "quartile": index + 1,
                "margin_bounds": [float(quartiles[index]), float(quartiles[index + 1])],
                "count": int(np.sum(mask)),
                "mean_error": float(np.mean(error[mask])),
            }
        )

    inverse_rows = retained["inverse_closure"]["rows"]
    by_vowel = {}
    for vowel in ("a", "e", "i", "o", "u"):
        rows = [row for row in inverse_rows if row["expected_vowel"] == vowel]
        by_vowel[vowel] = {
            "mean_acoustic_cycle_distance": float(
                np.mean([row["acoustic_cycle_distance"] for row in rows])
            ),
            "mean_projected_motor_error": float(
                np.mean([row["projected_motor_error"] for row in rows])
            ),
            "mean_formant_errors_hz": np.mean(
                [row["formant_errors_hz"] for row in rows], axis=0
            ).tolist(),
            "agreement_fraction": float(
                np.mean([row["combined_agreement"] for row in rows])
            ),
        }

    report = {
        "mode": "h4_option_g_failure_analysis_post_hoc",
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "candidate_configuration_changed": False,
        "threshold_selected": False,
        "spent_forward_points_replayed_descriptively": True,
        "correlations": {
            "error_vs_nearest_prototype_motor_distance": float(
                np.corrcoef(error, nearest_distances)[0, 1]
            ),
            "error_vs_boundary_margin": float(
                np.corrcoef(error, margin)[0, 1]
            ),
            "error_vs_constriction_magnitude": float(
                np.corrcoef(error, constrictions)[0, 1]
            ),
        },
        "boundary_margin_quartiles": margin_rows,
        "inverse_by_vowel": by_vowel,
        "interpretation": "Hard winner boundaries and high-constriction curvature both contribute. This supports testing a smooth local partition-of-unity mechanism on disjoint points, not increasing Option G capacity or relaxing its gates.",
        "recorded_audio_accessed": False,
        "protected_data_accessed": False,
        "promotion_allowed": False,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
