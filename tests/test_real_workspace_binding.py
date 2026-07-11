from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cora_infans.real_workspace_binding import (
    BoundedEfferenceWorkspace,
    evaluate_real_workspace_binding_design,
)


FAMILIES = (
    "m2_calibrated_matrix_20260429_092841",
    "m2_energy_onset_matrix_20260429_093045",
    "m2_energy_forward_tol75_matrix_20260429_093302",
)


def _write_clip(root, family, kind, seed, *, self_score, speech_ratio) -> None:
    path = root / family / f"{kind}_seed{seed}"
    path.mkdir(parents=True)
    np.save(path / "observed.npy", np.asarray([seed, self_score, speech_ratio]))
    (path / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "self_score": self_score,
                "speech_ratio": speech_ratio,
                "self_only": kind == "self_only",
                "world_only": False,
            }
        )
    )


def _write_fixture(root) -> None:
    for family in FAMILIES:
        for seed in range(3):
            _write_clip(
                root,
                family,
                "self_only",
                seed,
                self_score=0.30 + seed * 0.01,
                speech_ratio=0.0,
            )
            _write_clip(
                root,
                family,
                "mixed",
                seed,
                self_score=0.28 + seed * 0.01,
                speech_ratio=0.90,
            )
    control = root / "control_world-only_seed0_in3_out5_noise4_20260429_091828"
    control.mkdir()
    np.save(control / "observed.npy", np.asarray([0.0, 1.0]))
    (control / "summary.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "self_score": 0.0,
                "speech_ratio": 1.0,
                "self_only": False,
                "world_only": True,
            }
        )
    )


def test_bounded_workspace_evicts_oldest_binding() -> None:
    workspace = BoundedEfferenceWorkspace(capacity=2)
    workspace.push("a")
    workspace.push("b")
    workspace.push("c")

    assert workspace.snapshot() == ("b", "c")
    assert workspace.contains("a") is False
    assert workspace.contains("c") is True


def test_real_workspace_design_exposes_collision_and_capacity(tmp_path) -> None:
    _write_fixture(tmp_path)

    report = evaluate_real_workspace_binding_design(root=tmp_path)

    selected = report["capacity_sweep"]["3"]
    assert report["promotion_allowed"] is False
    assert report["all_design_criteria_passed"] is True
    assert selected["identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert selected["identity_only"]["metrics"]["world_recall"] < 1.0
    assert (
        report["capacity_sweep"]["1"]["identity_plus_world_evidence"]["metrics"][
            "f1"
        ]
        < 1.0
    )


def test_retained_real_workspace_design_is_non_promoting() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-real-workspace-binding-design-audit.json"
        ).read_text()
    )

    assert report["promotion_allowed"] is False
    assert report["all_design_criteria_passed"] is True
    selected = report["capacity_sweep"]["3"]
    assert selected["flat"]["metrics"]["f1"] < 0.80
    assert selected["identity_only"]["metrics"]["world_recall"] == 0.25
    assert selected["identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert len(report["source_observed_sha256"]) == 19
    assert all(len(value) == 64 for value in report["source_observed_sha256"].values())
