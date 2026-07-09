"""Reward terms for the simulation-only M8 speaking-policy ladder."""

from cora_infans.speaking_policy import ConversationFeatures
from cora_infans.speaking_reward import score_speaking_action


def test_repair_after_collision_is_rewarded() -> None:
    reward = score_speaking_action(
        action="repair",
        features=ConversationFeatures(was_collided=True, was_interrupted=True, has_deferred_output=True),
    )

    assert reward["total"] == 1.5
    assert reward["terms"]["successful_repair"] == 1.5


def test_missing_repair_after_collision_is_penalized() -> None:
    reward = score_speaking_action(
        action="wait",
        features=ConversationFeatures(was_collided=True, was_interrupted=True, has_deferred_output=True),
    )

    assert reward["total"] == -1.5
    assert reward["terms"]["missed_repair"] == -1.5


def test_yield_during_active_collision_is_rewarded_as_interruption_response() -> None:
    reward = score_speaking_action(
        action="yield",
        features=ConversationFeatures(is_agent_speaking=True, was_collided=True, was_interrupted=True),
    )

    assert reward["terms"]["successful_yield"] == 1.0
    assert "missed_repair" not in reward["terms"]


def test_clarification_when_needed_is_rewarded_and_unneeded_clarification_penalized() -> None:
    needed = score_speaking_action(
        action="clarify",
        features=ConversationFeatures(needs_clarification=True),
    )
    unneeded = score_speaking_action(
        action="clarify",
        features=ConversationFeatures(needs_clarification=False),
    )

    assert needed["terms"]["successful_clarify"] == 1.0
    assert unneeded["terms"]["unnecessary_clarify"] == -0.5


def test_smooth_turn_transition_is_rewarded_for_speak_after_short_silence() -> None:
    reward = score_speaking_action(
        action="speak",
        features=ConversationFeatures(queued_output=True, silence_ms=1200),
    )

    assert reward["terms"]["smooth_turn_transition"] == 0.5
    assert reward["total"] == 0.5


def test_stalled_conversation_penalty_scales_after_three_seconds() -> None:
    reward = score_speaking_action(
        action="wait",
        features=ConversationFeatures(queued_output=True, silence_ms=5500),
    )

    assert reward["terms"]["stalled_conversation"] == -0.5
    assert reward["total"] == -0.5


def test_speaking_into_an_unrepaired_collision_is_penalized() -> None:
    # Collision flagged without the agent having spoken/deferred output:
    # speaking straight into it must not be rewarded as a smooth turn.
    reward = score_speaking_action(
        action="speak",
        features=ConversationFeatures(was_collided=True, was_interrupted=True, queued_output=True, silence_ms=2200),
    )

    assert reward["terms"]["missed_repair"] == -1.5
    assert "smooth_turn_transition" not in reward["terms"]
    assert reward["total"] == -1.5


def test_smooth_turn_transition_at_policy_speak_threshold() -> None:
    # The policy default speaks at >=2000ms silence; the bonus must cover that
    # normal turn handoff rather than being capped below the speak threshold.
    reward = score_speaking_action(
        action="speak",
        features=ConversationFeatures(queued_output=True, silence_ms=2000),
    )

    assert reward["terms"]["smooth_turn_transition"] == 0.5
    assert reward["total"] == 0.5
