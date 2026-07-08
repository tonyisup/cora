"""Tests for the ActiveAttentionGovernor (pure logic, no hardware)."""

from cora_infans.io.active_attention import (
    ActiveAttentionGovernor,
    AttentionConfig,
    DEFAULT_SALIENCY_GATE,
    derive_target_azimuth,
)


def _governor(deadband=5.0, slew=180.0, gate=DEFAULT_SALIENCY_GATE):
    return ActiveAttentionGovernor(
        AttentionConfig(
            saliency_gate=gate, deadband_deg=deadband, max_slew_deg_per_s=slew
        )
    )


def test_saliency_gate_blocks_quiet_frame():
    g = _governor()
    d = g.propose(30.0, saliency_score=0.01, now_ms=0)
    assert d.acted is False
    assert d.reason == "below_saliency_gate"
    assert g.steer_count == 0


def test_first_salient_target_steers():
    g = _governor()
    d = g.propose(40.0, saliency_score=0.05, now_ms=0)
    assert d.acted is True
    assert d.reason == "steer"
    g.register_steer(40.0, now_ms=0)
    assert g.current_azimuth_deg == 40.0
    assert g.steer_count == 1


def test_deadband_prevents_jitter():
    g = _governor(deadband=5.0)
    g.propose(40.0, 0.05, now_ms=0)
    g.register_steer(40.0, now_ms=0)
    # 3 deg move -> within deadband -> no steer
    d = g.propose(43.0, 0.05, now_ms=200)
    assert d.acted is False
    assert d.reason == "within_deadband"
    # 10 deg move -> exceeds deadband -> steer
    d2 = g.propose(50.0, 0.05, now_ms=400)
    assert d2.acted is True
    g.register_steer(50.0, now_ms=400)
    assert g.steer_count == 2


def test_slew_limit_blocks_unbiological_sweep():
    g = _governor(deadband=1.0, slew=180.0)
    g.propose(0.0, 0.05, now_ms=0)
    g.register_steer(0.0, now_ms=0)
    # 180 deg in 100 ms => 1800 deg/s >> 180 limit => blocked
    d = g.propose(180.0, 0.05, now_ms=100)
    assert d.acted is False
    assert d.reason == "slew_limited"
    # 50 deg in 1000 ms => 50 deg/s <= 180 => allowed
    d2 = g.propose(50.0, 0.05, now_ms=1000)
    assert d2.acted is True


def test_reward_accumulates_on_salient():
    g = _governor()
    g.reward(0.05)
    g.reward(0.07)
    assert g.reward_count == 2
    assert abs(g.reward_sum - 0.12) < 1e-9


def test_derive_target_azimuth_lr_resolution():
    # Only 2 channels -> coarse L/R, not 360.
    assert derive_target_azimuth([0.0, 0.0]) == 0.0
    # Right louder -> positive azimuth
    assert derive_target_azimuth([0.1, 0.3]) > 0.0
    # Left louder -> negative azimuth
    assert derive_target_azimuth([0.3, 0.1]) < 0.0
    # Single channel -> straight ahead
    assert derive_target_azimuth([0.5]) == 0.0


def test_summary_reports_state():
    g = _governor()
    g.propose(30.0, 0.05, now_ms=0)
    g.register_steer(30.0, now_ms=0)
    s = g.summary()
    assert s["current_azimuth_deg"] == 30.0
    assert s["steer_count"] == 1
    assert len(s["history"]) == 1


def test_explicit_external_target_overrides_derived_estimate():
    # When an external source of truth supplies a bearing, the governor steers
    # toward THAT, not the coarse derived L/R estimate. Use a realistic interval
    # so the 90 deg saccade stays under the 180 deg/s slew cap (>= 500 ms).
    g = _governor(deadband=5.0)
    g.propose(0.0, 0.05, now_ms=0)  # first steer (derived est = 0)
    g.register_steer(0.0, now_ms=0)
    # External geometry says the speaker is at 270 (left); derived L/R says ~10.
    d = g.propose(10.0, 0.05, now_ms=600, explicit_target_deg=270.0)
    assert d.acted is True
    assert d.proposed_azimuth_deg == 270.0  # explicit bearing wins
    g.register_steer(270.0, now_ms=600)
    assert g.current_azimuth_deg == 270.0


def test_explicit_external_target_none_falls_back_to_derived():
    g = _governor(deadband=5.0)
    d = g.propose(40.0, 0.05, now_ms=0, explicit_target_deg=None)
    assert d.acted is True
    assert d.proposed_azimuth_deg == 40.0  # derived estimate used
