"""Combined simulation-only readiness gate before any live speaking work."""

from cora_infans.speaking_simulator import run_speaking_readiness_gate


def test_speaking_readiness_gate_requires_both_simulation_promotions() -> None:
    report = run_speaking_readiness_gate()

    assert report["mode"] == "simulation_only_speaking_readiness_gate"
    assert report["verdict"] == "promote_simulation_only"
    assert report["simulated_speak_authorized"] is True
    assert report["live_playback"] is False
    assert report["live_playback_allowed"] is False
    assert report["criteria"]["speaking_policy_promoted"]["passed"] is True
    assert report["criteria"]["self_forward_model_promoted"]["passed"] is True
    assert report["criteria"]["live_playback_blocked"]["passed"] is True
    assert report["speaking_policy_gate"]["verdict"] == "promote_simulation_only"
    assert report["self_forward_model_gate"]["verdict"] == "promote_simulation_only"


def test_speaking_readiness_gate_holds_when_speaking_policy_gate_holds() -> None:
    report = run_speaking_readiness_gate(min_speaking_reward_ratio=1.1)

    assert report["verdict"] == "hold"
    assert report["simulated_speak_authorized"] is False
    assert report["criteria"]["speaking_policy_promoted"]["passed"] is False
    assert report["criteria"]["self_forward_model_promoted"]["passed"] is True
    assert report["live_playback_allowed"] is False


def test_speaking_readiness_gate_holds_when_self_forward_model_gate_holds() -> None:
    report = run_speaking_readiness_gate(min_self_world_correlation=1.1)

    assert report["verdict"] == "hold"
    assert report["simulated_speak_authorized"] is False
    assert report["criteria"]["speaking_policy_promoted"]["passed"] is True
    assert report["criteria"]["self_forward_model_promoted"]["passed"] is False
    assert report["live_playback_allowed"] is False
