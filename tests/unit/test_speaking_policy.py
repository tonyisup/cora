"""Unit tests for M8 SpeakingPolicy."""

from __future__ import annotations

import pytest

from pfc.speaking_policy import (
    HeuristicSpeakingPolicy,
    SpeakingPolicy,
    ACTION_NAMES,
)


def make_features(
    conversation_state: str = "idle",
    is_agent_speaking: bool = False,
    is_agent_deferred: bool = False,
    is_user_speaking: bool = False,
    was_user_interrupted: bool = False,
    seconds_since_last_user_speech: float = 999.0,
    seconds_since_last_agent_speech: float = 999.0,
    seconds_agent_has_been_speaking: float = 0.0,
    seconds_user_has_been_speaking: float = 0.0,
    total_turns: int = 0,
    repair_count: int = 0,
    clarification_count: int = 0,
    yield_count: int = 0,
    just_interrupted: bool = False,
    just_collided: bool = False,
    just_deferred: bool = False,
    just_resumed: bool = False,
    has_queued_output: bool = False,
) -> dict:
    return {
        "conversation_state": conversation_state,
        "is_agent_speaking": is_agent_speaking,
        "is_agent_deferred": is_agent_deferred,
        "is_user_speaking": is_user_speaking,
        "was_user_interrupted": was_user_interrupted,
        "seconds_since_last_user_speech": seconds_since_last_user_speech,
        "seconds_since_last_agent_speech": seconds_since_last_agent_speech,
        "seconds_agent_has_been_speaking": seconds_agent_has_been_speaking,
        "seconds_user_has_been_speaking": seconds_user_has_been_speaking,
        "total_turns": total_turns,
        "repair_count": repair_count,
        "clarification_count": clarification_count,
        "yield_count": yield_count,
        "just_interrupted": just_interrupted,
        "just_collided": just_collided,
        "just_deferred": just_deferred,
        "just_resumed": just_resumed,
        "has_queued_output": has_queued_output,
    }


class TestHeuristicSpeakingPolicySpeak:
    def test_speak_after_silence_with_queued_output(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            has_queued_output=True,
            seconds_since_last_user_speech=3.0,
            seconds_since_last_agent_speech=999.0,
        )
        assert policy.decide(f) == "speak"

    def test_wait_when_no_queued_output(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(has_queued_output=False)
        assert policy.decide(f) == "wait"

    def test_wait_when_silence_too_short(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            has_queued_output=True,
            seconds_since_last_user_speech=0.5,
        )
        assert policy.decide(f) == "wait"


class TestHeuristicSpeakingPolicyYield:
    def test_yield_on_interruption(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            just_interrupted=True,
            is_agent_speaking=True,
            is_user_speaking=True,
        )
        assert policy.decide(f) == "yield"

    def test_yield_when_both_speaking(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            is_agent_speaking=True,
            is_user_speaking=True,
        )
        assert policy.decide(f) == "yield"


class TestHeuristicSpeakingPolicyWait:
    def test_wait_while_user_speaking(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            is_user_speaking=True,
            has_queued_output=True,
            seconds_since_last_user_speech=999.0,
        )
        assert policy.decide(f) == "wait"

    def test_wait_when_user_speaking_no_deferred(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(is_user_speaking=True)
        assert policy.decide(f) == "wait"


class TestHeuristicSpeakingPolicyRepair:
    def test_repair_after_collision(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(just_collided=True)
        assert policy.decide(f) == "repair"

    def test_repair_when_in_repair_state(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(conversation_state="repair")
        assert policy.decide(f) == "repair"


class TestHeuristicSpeakingPolicyResume:
    def test_resume_deferred_output(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            is_agent_deferred=True,
            is_user_speaking=False,
        )
        assert policy.decide(f) == "speak"


class TestHeuristicSpeakingPolicyLongMonologue:
    def test_yield_after_long_monologue(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            is_agent_speaking=True,
            seconds_agent_has_been_speaking=15.0,
        )
        assert policy.decide(f) == "yield"

    def test_keep_speaking_at_normal_duration(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features(
            is_agent_speaking=True,
            seconds_agent_has_been_speaking=3.0,
        )
        assert policy.decide(f) == "speak"


class TestHeuristicSpeakingPolicyDefault:
    def test_default_is_wait(self):
        policy = HeuristicSpeakingPolicy()
        f = make_features()
        assert policy.decide(f) == "wait"

    def test_reset(self):
        policy = HeuristicSpeakingPolicy()
        policy.reset()  # should be a no-op, no error


class TestHeuristicSpeakingPolicyActionsValid:
    def test_all_decisions_return_valid_action_names(self):
        """Fuzz test: ensure policy never returns an invalid action."""
        policy = HeuristicSpeakingPolicy()
        scenarios = [
            make_features(),
            make_features(has_queued_output=True, seconds_since_last_user_speech=999.0),
            make_features(is_user_speaking=True),
            make_features(is_agent_speaking=True),
            make_features(is_agent_speaking=True, is_user_speaking=True),
            make_features(just_interrupted=True),
            make_features(just_collided=True),
            make_features(is_agent_deferred=True),
            make_features(has_queued_output=True, seconds_since_last_user_speech=0.5),
            make_features(is_agent_speaking=True, seconds_agent_has_been_speaking=12.0),
        ]
        for f in scenarios:
            action = policy.decide(f)
            assert action in ACTION_NAMES, f"Invalid action '{action}' for features {f}"
