"""Promotion gate for the simulation-only M8 speaking-policy ladder."""

from cora_infans.speaking_simulator import run_speaking_policy_promotion_gate


def test_speaking_policy_promotion_gate_promotes_canonical_episodes() -> None:
    report = run_speaking_policy_promotion_gate()

    assert report["mode"] == "simulation_only_speaking_policy_promotion_gate"
    assert report["verdict"] == "promote_simulation_only"
    assert report["live_playback"] is False
    assert report["live_playback_allowed"] is False
    assert report["summary"]["episode_count"] >= 4
    assert all(criterion["passed"] for criterion in report["criteria"].values())
    assert report["criteria"]["reward_threshold"]["value"] >= 0.8
    assert report["criteria"]["overlap_yields"]["value"] == 1.0
    assert report["criteria"]["no_false_repairs"]["value"] == 1.0
    assert report["criteria"]["no_unnecessary_clarification"]["value"] == 1.0
    assert report["criteria"]["no_excessive_stall_penalty"]["value"] == 1.0


def test_speaking_policy_promotion_gate_holds_when_reward_threshold_is_impossible() -> None:
    report = run_speaking_policy_promotion_gate(min_reward_ratio=1.1)

    assert report["verdict"] == "hold"
    assert report["criteria"]["reward_threshold"]["passed"] is False
    assert report["live_playback_allowed"] is False
