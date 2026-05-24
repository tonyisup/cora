"""Unit tests for M8 reward function."""

from __future__ import annotations

import pytest

from scripts.m8_reward import M8RewardConfig, M8RewardSignal, compute_m8_reward


def make_features(
    is_user_speaking: bool = False,
    is_agent_speaking: bool = False,
    is_agent_deferred: bool = False,
    just_interrupted: bool = False,
    just_collided: bool = False,
    just_resumed: bool = False,
    seconds_since_last_user_speech: float = 999.0,
    seconds_agent_has_been_speaking: float = 0.0,
    conversation_state: str = "idle",
    has_queued_output: bool = False,
) -> dict:
    return {
        "conversation_state": conversation_state,
        "is_agent_speaking": is_agent_speaking,
        "is_agent_deferred": is_agent_deferred,
        "is_user_speaking": is_user_speaking,
        "was_user_interrupted": False,
        "seconds_since_last_user_speech": seconds_since_last_user_speech,
        "seconds_since_last_agent_speech": 999.0,
        "seconds_agent_has_been_speaking": seconds_agent_has_been_speaking,
        "seconds_user_has_been_speaking": 0.0,
        "total_turns": 0,
        "repair_count": 0,
        "clarification_count": 0,
        "yield_count": 0,
        "just_interrupted": just_interrupted,
        "just_collided": just_collided,
        "just_deferred": False,
        "just_resumed": just_resumed,
        "has_queued_output": has_queued_output,
    }


class TestM8RewardSuccessfulRepair:
    def test_repair_then_user_speaks(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True, just_collided=True)
        after = make_features(is_user_speaking=True, is_agent_speaking=False)
        signal = compute_m8_reward(config, "repair", before, after)
        assert signal.reward == config.successful_repair
        assert "successful_repair" in signal.components

    def test_repair_no_user_response(self):
        config = M8RewardConfig()
        before = make_features(just_collided=True)
        after = make_features(seconds_since_last_user_speech=999.0)
        signal = compute_m8_reward(config, "repair", before, after)
        # No positive signal if user doesn't respond
        assert signal.reward <= 0.0 or "successful_repair" not in signal.components


class TestM8RewardSuccessfulClarify:
    def test_clarify_then_user_responds(self):
        config = M8RewardConfig()
        before = make_features(is_agent_deferred=True)
        after = make_features(is_user_speaking=True, is_agent_speaking=False)
        signal = compute_m8_reward(config, "clarify", before, after)
        assert signal.reward == config.successful_clarify
        assert "successful_clarify" in signal.components

    def test_unnecessary_clarify(self):
        config = M8RewardConfig()
        before = make_features(is_agent_deferred=True)
        after = make_features(seconds_since_last_user_speech=999.0)
        signal = compute_m8_reward(config, "clarify", before, after)
        assert signal.reward == config.unnecessary_clarify
        assert "unnecessary_clarify" in signal.components


class TestM8RewardSmoothTurn:
    def test_smooth_turn_transition(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True)
        after = make_features(is_user_speaking=True, is_agent_speaking=False,
                              seconds_since_last_user_speech=0.5)
        signal = compute_m8_reward(config, "wait", before, after)
        assert signal.reward == config.smooth_turn_transition
        assert "smooth_turn_transition" in signal.components


class TestM8RewardResumeAfterYield:
    def test_resume_gives_reward(self):
        config = M8RewardConfig()
        before = make_features(is_agent_deferred=True)
        after = make_features(just_resumed=True, is_agent_speaking=True)
        signal = compute_m8_reward(config, "speak", before, after)
        assert signal.reward == config.resume_after_yield
        assert "resume_after_yield" in signal.components


class TestM8RewardSelfInterrupt:
    def test_yield_without_user_speech(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True, is_user_speaking=False)
        after = make_features(is_agent_speaking=False)
        signal = compute_m8_reward(config, "yield", before, after)
        assert signal.reward == config.self_interrupt
        assert "self_interrupt" in signal.components

    def test_yield_with_user_speaking_no_penalty(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True, is_user_speaking=True,
                                just_interrupted=True)
        after = make_features(is_agent_speaking=False)
        signal = compute_m8_reward(config, "yield", before, after)
        # Self-interrupt should NOT fire when user was actually speaking
        assert "self_interrupt" not in signal.components


class TestM8RewardStall:
    def test_stall_penalty(self):
        config = M8RewardConfig()
        before = make_features(has_queued_output=True)
        after = make_features(
            has_queued_output=True,
            is_user_speaking=False,
            is_agent_speaking=False,
            seconds_since_last_user_speech=5.0,
        )
        signal = compute_m8_reward(config, "wait", before, after)
        assert signal.reward < 0.0
        assert "stall" in signal.components

    def test_no_stall_when_speaking(self):
        config = M8RewardConfig()
        before = make_features(has_queued_output=True, is_agent_speaking=True)
        after = make_features(is_agent_speaking=True, seconds_since_last_user_speech=5.0)
        signal = compute_m8_reward(config, "speak", before, after)
        assert "stall" not in signal.components


class TestM8RewardLongMonologue:
    def test_monologue_penalty(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True,
                                seconds_agent_has_been_speaking=10.0)
        after = make_features(is_agent_speaking=True,
                               seconds_agent_has_been_speaking=12.0)
        signal = compute_m8_reward(config, "speak", before, after)
        assert signal.reward < 0.0
        assert "long_monologue" in signal.components

    def test_no_monologue_penalty_at_normal_duration(self):
        config = M8RewardConfig()
        before = make_features(is_agent_speaking=True,
                                seconds_agent_has_been_speaking=3.0)
        after = make_features(is_agent_speaking=True,
                               seconds_agent_has_been_speaking=4.0)
        signal = compute_m8_reward(config, "speak", before, after)
        assert "long_monologue" not in signal.components


class TestM8RewardNoSignal:
    def test_idle_wait_no_reward(self):
        config = M8RewardConfig()
        before = make_features()
        after = make_features()
        signal = compute_m8_reward(config, "wait", before, after)
        assert signal.reward == 0.0
        assert signal.reason == "no_m8_signal"


class TestM8RewardComponents:
    def test_components_sum_to_total(self):
        config = M8RewardConfig()
        before = make_features(
            is_agent_speaking=True,
            just_collided=True,
        )
        after = make_features(
            is_user_speaking=True,
            is_agent_speaking=False,
            just_resumed=False,
        )
        signal = compute_m8_reward(config, "repair", before, after)
        expected = sum(signal.components.values())
        assert signal.reward == pytest.approx(expected, abs=1e-6)
