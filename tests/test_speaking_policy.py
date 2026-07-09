"""Simulation-only speaking policy adapted from og-cora M8.

These tests define the safe policy layer before any live playback exists.
"""

from cora_infans.speaking_policy import ConversationFeatures, HeuristicSpeakingPolicy


def test_user_speaking_waits() -> None:
    policy = HeuristicSpeakingPolicy()
    features = ConversationFeatures(is_user_speaking=True, queued_output=True, silence_ms=3000)

    assert policy.decide(features) == "wait"


def test_agent_collision_yields() -> None:
    policy = HeuristicSpeakingPolicy()
    features = ConversationFeatures(is_agent_speaking=True, was_collided=True, queued_output=True)

    assert policy.decide(features) == "yield"


def test_interrupted_agent_with_deferred_output_repairs() -> None:
    policy = HeuristicSpeakingPolicy()
    features = ConversationFeatures(was_interrupted=True, has_deferred_output=True)

    assert policy.decide(features) == "repair"


def test_clarification_need_clarifies() -> None:
    policy = HeuristicSpeakingPolicy()
    features = ConversationFeatures(needs_clarification=True)

    assert policy.decide(features) == "clarify"


def test_queued_output_after_silence_speaks() -> None:
    policy = HeuristicSpeakingPolicy(min_silence_before_speak_ms=2000)
    features = ConversationFeatures(queued_output=True, silence_ms=2500)

    assert policy.decide(features) == "speak"


def test_no_queued_output_waits() -> None:
    policy = HeuristicSpeakingPolicy()
    features = ConversationFeatures(silence_ms=5000)

    assert policy.decide(features) == "wait"


def test_policy_reset_is_available_and_side_effect_free() -> None:
    policy = HeuristicSpeakingPolicy()

    assert policy.reset() is None
    assert policy.decide(ConversationFeatures(queued_output=True, silence_ms=2500)) == "speak"
