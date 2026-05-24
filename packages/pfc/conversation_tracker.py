"""PFC — Conversation Tracker for M8 speaking policy.

Observes frame logs and maintains a conversation state machine.
Extracts features for the speaking policy to make decisions on.

Same pattern as M6 SelfModel: reads frame logs after the fact,
maintains stateful view of the conversation. Does NOT gate or control.

Usage:
    tracker = ConversationTracker()
    for frame in frame_log_iterator:
        tracker.process_frame(frame)

    features = tracker.get_features()
    # features is a dict of bools/numerics for the SpeakingPolicy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ConversationState(Enum):
    """High-level conversation state."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    REPAIR = "repair"
    CLARIFICATION = "clarification"


@dataclass
class ConversationFeatures:
    """Features extracted from the conversation state for the SpeakingPolicy.

    All fields are simple types for easy serialization and policy input.
    """
    # Current state
    conversation_state: str = "idle"  # ConversationState.value

    # Agent state
    is_agent_speaking: bool = False
    is_agent_deferred: bool = False   # has deferred output waiting

    # User state
    is_user_speaking: bool = False
    was_user_interrupted: bool = False  # user started but was cut off

    # Timing (seconds)
    seconds_since_last_user_speech: float = 0.0
    seconds_since_last_agent_speech: float = 0.0
    seconds_agent_has_been_speaking: float = 0.0
    seconds_user_has_been_speaking: float = 0.0

    # Counts (within current conversation)
    total_turns: int = 0
    repair_count: int = 0
    clarification_count: int = 0
    yield_count: int = 0

    # Recent events (cleared after policy decision)
    just_interrupted: bool = False      # overlap detected this step
    just_collided: bool = False          # collision repair triggered this step
    just_deferred: bool = False          # agent deferred this step
    just_resumed: bool = False           # agent resumed deferred output

    # Queue state (what the agent wants to say)
    has_queued_output: bool = False


class ConversationTracker:
    """Tracks conversation state from frame logs.

    Process frames one at a time via process_frame(). After each frame,
    call get_features() to get the current feature dict for policy decisions.

    The stateless per-frame fields (just_interrupted, just_collided, etc.)
    are cleared after each feature extraction to avoid double-counting.
    """

    def __init__(self) -> None:
        self._state = ConversationState.IDLE

        # Timing
        self._last_user_speech_time: float = 0.0
        self._last_agent_speech_time: float = 0.0
        self._agent_speak_start_time: float = -1.0  # -1 = not speaking
        self._user_speak_start_time: float = -1.0   # -1 = not speaking
        self._current_time: float = 0.0
        self._agent_has_spoken: bool = False  # set True once agent starts speaking

        # Counts
        self._total_turns: int = 0
        self._repair_count: int = 0
        self._clarification_count: int = 0
        self._yield_count: int = 0

        # Flags (cleared after get_features)
        self._just_interrupted: bool = False
        self._just_collided: bool = False
        self._just_deferred: bool = False
        self._just_resumed: bool = False

        # Deferred output tracking
        self._is_agent_deferred: bool = False
        self._has_queued_output: bool = False

    @property
    def state(self) -> ConversationState:
        return self._state

    def reset(self) -> None:
        """Reset all state for a new conversation."""
        self.__init__()

    def process_frame(self, frame: Dict[str, Any]) -> None:
        """Process a single frame log entry.

        Parameters
        ----------
        frame : dict
            A StreamingFrameLog-like dict with at minimum:
            - timestamp_s: float
            - verdict: str (SILENCE, SELF, WORLD, BOTH)
            - controller_state: str (IDLE, LISTEN, SPEAK)
            - tts_playing: bool
            - deferred: bool
            - resumed: bool
            - should_interrupt: bool
        """
        self._current_time = frame.get("timestamp_s", self._current_time)
        verdict = frame.get("verdict", "SILENCE")
        controller_state = frame.get("controller_state", "IDLE")
        tts_playing = frame.get("tts_playing", False)
        deferred = frame.get("deferred", False)
        resumed = frame.get("resumed", False)
        should_interrupt = frame.get("should_interrupt", False)

        # Track user speech (WORLD verdict)
        if verdict == "WORLD":
            if self._user_speak_start_time < 0:
                self._user_speak_start_time = self._current_time
            self._last_user_speech_time = self._current_time

        # Track agent speech (SELF or BOTH verdicts while TTS is playing)
        is_agent_speaking_now = tts_playing and verdict in ("SELF", "BOTH")
        if is_agent_speaking_now:
            if self._agent_speak_start_time < 0:
                self._agent_speak_start_time = self._current_time
                self._agent_has_spoken = True
            self._last_agent_speech_time = self._current_time

        # Detect transitions
        if should_interrupt and is_agent_speaking_now:
            self._just_interrupted = True
            self._yield_count += 1

        if deferred:
            self._just_deferred = True
            self._is_agent_deferred = True

        if resumed:
            self._just_resumed = True
            self._is_agent_deferred = False

        # Track repair (collision repair text present)
        if frame.get("collision_repair_text") is not None:
            self._just_collided = True
            self._repair_count += 1

        # Track clarification (controller in LISTEN with deferred output)
        if controller_state == "LISTEN" and self._is_agent_deferred:
            pass  # waiting for user to respond

        # Update conversation state machine
        self._update_state(controller_state, verdict, tts_playing)

        # Track turns (user starts speaking after agent finishes)
        if (verdict == "WORLD" and
                self._state == ConversationState.LISTENING and
                self._agent_has_spoken):
            self._total_turns += 1
            self._agent_has_spoken = False  # reset for next turn

        # Reset per-frame flags when speech ends
        if verdict != "WORLD":
            self._user_speak_start_time = -1.0
        if not is_agent_speaking_now:
            self._agent_speak_start_time = -1.0

    def _update_state(
        self,
        controller_state: str,
        verdict: str,
        tts_playing: bool,
    ) -> None:
        """Update the conversation state machine."""
        if controller_state == "SPEAK" and tts_playing:
            self._state = ConversationState.SPEAKING
        elif controller_state == "LISTEN":
            if self._is_agent_deferred:
                # Agent is waiting to resume after yielding
                self._state = ConversationState.LISTENING
            else:
                self._state = ConversationState.LISTENING
        elif controller_state == "IDLE":
            if self._just_collided:
                self._state = ConversationState.REPAIR
            elif self._just_interrupted:
                self._state = ConversationState.LISTENING
            else:
                self._state = ConversationState.IDLE

    def get_features(self) -> Dict[str, Any]:
        """Extract current conversation features for the SpeakingPolicy.

        Clears the per-frame edge-triggered flags after extraction.
        """
        dt = self._current_time

        features = {
            "conversation_state": self._state.value,
            "is_agent_speaking": self._state == ConversationState.SPEAKING,
            "is_agent_deferred": self._is_agent_deferred,
            "is_user_speaking": self._user_speak_start_time >= 0,
            "was_user_interrupted": self._just_interrupted,
            "seconds_since_last_user_speech": (
                dt - self._last_user_speech_time if self._last_user_speech_time > 0 else 999.0
            ),
            "seconds_since_last_agent_speech": (
                dt - self._last_agent_speech_time if self._last_agent_speech_time > 0 else 999.0
            ),
            "seconds_agent_has_been_speaking": (
                dt - self._agent_speak_start_time if self._agent_speak_start_time >= 0 else 0.0
            ),
            "seconds_user_has_been_speaking": (
                dt - self._user_speak_start_time if self._user_speak_start_time >= 0 else 0.0
            ),
            "total_turns": self._total_turns,
            "repair_count": self._repair_count,
            "clarification_count": self._clarification_count,
            "yield_count": self._yield_count,
            "just_interrupted": self._just_interrupted,
            "just_collided": self._just_collided,
            "just_deferred": self._just_deferred,
            "just_resumed": self._just_resumed,
            "has_queued_output": self._has_queued_output,
        }

        # Clear edge-triggered flags
        self._just_interrupted = False
        self._just_collided = False
        self._just_deferred = False
        self._just_resumed = False

        return features
