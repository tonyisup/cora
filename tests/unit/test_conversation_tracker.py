"""Unit tests for M8 ConversationTracker."""

from __future__ import annotations

import pytest

from pfc.conversation_tracker import (
    ConversationFeatures,
    ConversationState,
    ConversationTracker,
)


def make_frame(
    timestamp_s: float = 0.0,
    verdict: str = "SILENCE",
    controller_state: str = "IDLE",
    tts_playing: bool = False,
    deferred: bool = False,
    resumed: bool = False,
    should_interrupt: bool = False,
    collision_repair_text: str | None = None,
    frame_index: int = 0,
) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "vad_speech": verdict != "SILENCE",
        "verdict": verdict,
        "controller_state": controller_state,
        "can_speak": controller_state in ("IDLE", "LISTEN"),
        "should_interrupt": should_interrupt,
        "tts_playing": tts_playing,
        "tts_position_s": 0.0,
        "tts_total_duration_s": 3.0,
        "output_source": "tts" if tts_playing else "silence",
        "repair_position_s": 0.0,
        "repair_total_duration_s": 0.0,
        "repair_queued": False,
        "repair_playing": False,
        "repair_completed": False,
        "m4_outcome": "not_attempted",
        "m4_confidence": None,
        "energy_db": -30.0 if verdict != "SILENCE" else -60.0,
        "self_match_score": 0.0,
        "self_match_score_per_frame": None,
        "is_in_motor_window": tts_playing,
        "deferred": deferred,
        "resumed": resumed,
        "collision_repair_text": collision_repair_text,
    }


class TestConversationTrackerInit:
    def test_initial_state_is_idle(self):
        tracker = ConversationTracker()
        assert tracker.state == ConversationState.IDLE

    def test_initial_features(self):
        tracker = ConversationTracker()
        f = tracker.get_features()
        assert f["conversation_state"] == "idle"
        assert f["is_agent_speaking"] is False
        assert f["is_user_speaking"] is False
        assert f["has_queued_output"] is False
        assert f["total_turns"] == 0

    def test_reset(self):
        tracker = ConversationTracker()
        frame = make_frame(verdict="WORLD", controller_state="LISTEN")
        tracker.process_frame(frame)
        tracker.reset()
        assert tracker.state == ConversationState.IDLE
        f = tracker.get_features()
        assert f["total_turns"] == 0


class TestConversationTrackerUserSpeech:
    def test_user_speaking_detected(self):
        tracker = ConversationTracker()
        frame = make_frame(timestamp_s=1.0, verdict="WORLD", controller_state="LISTEN")
        tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["is_user_speaking"] is True
        assert f["conversation_state"] == "listening"

    def test_user_speech_ends(self):
        tracker = ConversationTracker()
        tracker.process_frame(make_frame(timestamp_s=1.0, verdict="WORLD", controller_state="LISTEN"))
        tracker.get_features()  # clear flags
        tracker.process_frame(make_frame(timestamp_s=2.0, verdict="SILENCE", controller_state="IDLE"))
        f = tracker.get_features()
        assert f["is_user_speaking"] is False

    def test_seconds_since_user_speech(self):
        tracker = ConversationTracker()
        tracker.process_frame(make_frame(timestamp_s=1.0, verdict="WORLD", controller_state="LISTEN"))
        tracker.get_features()
        tracker.process_frame(make_frame(timestamp_s=3.0, verdict="SILENCE", controller_state="IDLE"))
        f = tracker.get_features()
        assert f["seconds_since_last_user_speech"] == pytest.approx(2.0, abs=0.1)


class TestConversationTrackerAgentSpeech:
    def test_agent_speaking_detected(self):
        tracker = ConversationTracker()
        frame = make_frame(
            timestamp_s=1.0, verdict="SELF", controller_state="SPEAK", tts_playing=True
        )
        tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["is_agent_speaking"] is True
        assert f["conversation_state"] == "speaking"

    def test_agent_speech_duration(self):
        tracker = ConversationTracker()
        for t in [0.0, 0.5, 1.0, 1.5]:
            frame = make_frame(
                timestamp_s=t, verdict="SELF", controller_state="SPEAK", tts_playing=True
            )
            tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["seconds_agent_has_been_speaking"] == pytest.approx(1.5, abs=0.1)


class TestConversationTrackerInterruption:
    def test_just_interrupted_flag(self):
        tracker = ConversationTracker()
        frame = make_frame(
            timestamp_s=1.0,
            verdict="BOTH",
            controller_state="SPEAK",
            tts_playing=True,
            should_interrupt=True,
        )
        tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["just_interrupted"] is True
        assert f["yield_count"] == 1

    def test_just_interrupted_cleared_after_get_features(self):
        tracker = ConversationTracker()
        frame = make_frame(
            timestamp_s=1.0,
            verdict="BOTH",
            controller_state="SPEAK",
            tts_playing=True,
            should_interrupt=True,
        )
        tracker.process_frame(frame)
        tracker.get_features()  # clears flags
        f2 = tracker.get_features()
        assert f2["just_interrupted"] is False


class TestConversationTrackerDeferred:
    def test_just_deferred_flag(self):
        tracker = ConversationTracker()
        frame = make_frame(timestamp_s=1.0, deferred=True)
        tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["just_deferred"] is True
        assert f["is_agent_deferred"] is True

    def test_resume_clears_deferred(self):
        tracker = ConversationTracker()
        tracker.process_frame(make_frame(timestamp_s=1.0, deferred=True))
        tracker.get_features()
        tracker.process_frame(make_frame(timestamp_s=2.0, resumed=True))
        f = tracker.get_features()
        assert f["just_resumed"] is True
        assert f["is_agent_deferred"] is False


class TestConversationTrackerCollision:
    def test_just_collided_flag(self):
        tracker = ConversationTracker()
        frame = make_frame(timestamp_s=1.0, collision_repair_text="Sorry, go ahead.")
        tracker.process_frame(frame)
        f = tracker.get_features()
        assert f["just_collided"] is True
        assert f["repair_count"] == 1


class TestConversationTrackerTurns:
    def test_turn_counted_when_user_starts_after_agent(self):
        tracker = ConversationTracker()
        # Agent speaks
        tracker.process_frame(make_frame(
            timestamp_s=1.0, verdict="SELF", controller_state="SPEAK", tts_playing=True
        ))
        tracker.get_features()
        # Agent finishes
        tracker.process_frame(make_frame(timestamp_s=2.0, verdict="SILENCE"))
        tracker.get_features()
        # User starts speaking
        tracker.process_frame(make_frame(
            timestamp_s=3.0, verdict="WORLD", controller_state="LISTEN"
        ))
        f = tracker.get_features()
        assert f["total_turns"] >= 1


class TestConversationTrackerEdgeTriggeredFlags:
    def test_all_edge_flags_cleared_after_get_features(self):
        tracker = ConversationTracker()
        frame = make_frame(
            timestamp_s=1.0,
            verdict="BOTH",
            controller_state="SPEAK",
            tts_playing=True,
            should_interrupt=True,
            deferred=True,
            resumed=True,
            collision_repair_text="Sorry.",
        )
        tracker.process_frame(frame)
        f1 = tracker.get_features()
        # All should be True in first call
        assert f1["just_interrupted"] is True
        assert f1["just_deferred"] is True
        assert f1["just_resumed"] is True
        assert f1["just_collided"] is True

        # All should be False in second call
        f2 = tracker.get_features()
        assert f2["just_interrupted"] is False
        assert f2["just_deferred"] is False
        assert f2["just_resumed"] is False
        assert f2["just_collided"] is False
