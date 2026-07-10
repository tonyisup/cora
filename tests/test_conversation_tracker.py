"""Deterministic conversation tracker for the M8-infans speaking-policy ladder."""

from cora_infans.conversation_tracker import ConversationEvent, ConversationTracker


def test_tracker_starts_idle_with_zeroed_features() -> None:
    tracker = ConversationTracker()

    features = tracker.get_conversation_features(now_ms=0)

    assert features.is_user_speaking is False
    assert features.is_agent_speaking is False
    assert features.queued_output is False
    assert features.has_deferred_output is False
    assert features.silence_ms == 0
    assert features.turn_count == 0


def test_user_speech_updates_speaking_state_and_turn_count() -> None:
    tracker = ConversationTracker()

    tracker.observe(ConversationEvent(timestamp_ms=100, kind="user_speech_start"))
    speaking = tracker.get_conversation_features(now_ms=150)
    tracker.observe(ConversationEvent(timestamp_ms=600, kind="user_speech_end"))
    quiet = tracker.get_conversation_features(now_ms=900)

    assert speaking.is_user_speaking is True
    assert speaking.turn_count == 1
    assert quiet.is_user_speaking is False
    assert quiet.silence_ms == 300


def test_agent_output_queue_and_speech_state() -> None:
    tracker = ConversationTracker()

    tracker.observe(ConversationEvent(timestamp_ms=0, kind="agent_output_queued", payload={"text": "hi"}))
    queued = tracker.get_conversation_features(now_ms=100)
    tracker.observe(ConversationEvent(timestamp_ms=200, kind="agent_speech_start"))
    speaking = tracker.get_conversation_features(now_ms=250)
    tracker.observe(ConversationEvent(timestamp_ms=700, kind="agent_speech_end"))
    done = tracker.get_conversation_features(now_ms=800)

    assert queued.queued_output is True
    assert speaking.is_agent_speaking is True
    assert speaking.queued_output is False
    assert done.is_agent_speaking is False
    assert done.silence_ms == 100


def test_overlap_marks_collision_interruption_and_deferred_output() -> None:
    tracker = ConversationTracker()

    tracker.observe(ConversationEvent(timestamp_ms=0, kind="agent_output_queued"))
    tracker.observe(ConversationEvent(timestamp_ms=100, kind="agent_speech_start"))
    tracker.observe(ConversationEvent(timestamp_ms=200, kind="overlap_detected"))

    features = tracker.get_conversation_features(now_ms=250)

    assert features.is_agent_speaking is True
    assert features.was_collided is True
    assert features.was_interrupted is True
    assert features.has_deferred_output is True


def test_repair_and_clarification_counts_are_tracked() -> None:
    tracker = ConversationTracker()

    tracker.observe(ConversationEvent(timestamp_ms=100, kind="repair_issued"))
    tracker.observe(ConversationEvent(timestamp_ms=200, kind="user_correction"))
    needs_clarification = tracker.get_conversation_features(now_ms=250)
    tracker.observe(ConversationEvent(timestamp_ms=300, kind="clarification_issued"))

    features = tracker.get_conversation_features(now_ms=400)

    assert features.repair_count == 1
    assert needs_clarification.needs_clarification is True
    assert features.needs_clarification is False
    assert features.clarification_count == 1


def test_repair_clears_deferred_output_state() -> None:
    tracker = ConversationTracker()
    tracker.observe(ConversationEvent(timestamp_ms=0, kind="agent_output_queued"))
    tracker.observe(ConversationEvent(timestamp_ms=100, kind="agent_speech_start"))
    tracker.observe(ConversationEvent(timestamp_ms=200, kind="overlap_detected"))

    tracker.observe(ConversationEvent(timestamp_ms=300, kind="repair_issued"))

    features = tracker.get_conversation_features(now_ms=300)
    assert features.has_deferred_output is False


def test_tracker_reset_clears_state() -> None:
    tracker = ConversationTracker()
    tracker.observe(ConversationEvent(timestamp_ms=100, kind="user_speech_start"))
    tracker.observe(ConversationEvent(timestamp_ms=200, kind="overlap_detected"))

    tracker.reset()
    features = tracker.get_conversation_features(now_ms=300)

    assert features.is_user_speaking is False
    assert features.was_collided is False
    assert features.turn_count == 0
