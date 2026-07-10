from cora_infans.workspace_ablation import run_workspace_ablation


def test_workspace_ablation_exposes_benefit_and_collision_failure() -> None:
    report = run_workspace_ablation()
    echo = report["scenarios"]["aligned_self_echo"]
    overlap = report["scenarios"]["legitimate_world_overlap"]

    assert echo["theta_gamma"]["metrics"]["f1"] > echo["flat"]["metrics"]["f1"]
    assert overlap["flat"]["metrics"]["recall"] == 1.0
    assert overlap["theta_gamma"]["metrics"]["recall"] == 1.0
    assert report["criterion_passed"] is True
    assert report["evidence_scope"] == "synthetic_known_timing_only"
