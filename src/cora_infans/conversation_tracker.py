"""Deterministic conversation tracker for simulation-only speaking policy.

This is the M8-infans observer layer: it converts scripted conversational events
into `ConversationFeatures` for `HeuristicSpeakingPolicy`. It has no audio I/O,
no playback, and no hardware access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .speaking_policy import ConversationFeatures

ConversationEventKind = Literal[
    "user_speech_start",
    "user_speech_end",
    "agent_output_queued",
    "agent_speech_start",
    "agent_speech_end",
    "agent_speech_yielded",
    "overlap_detected",
    "user_correction",
    "repair_issued",
    "clarification_issued",
]


@dataclass(frozen=True)
class ConversationEvent:
    timestamp_ms: int
    kind: ConversationEventKind
    payload: dict[str, object] = field(default_factory=dict)


class ConversationTracker:
    """Pure observer that maintains policy-facing conversation features."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._is_user_speaking = False
        self._is_agent_speaking = False
        self._queued_output = False
        self._has_deferred_output = False
        self._needs_clarification = False
        self._was_interrupted = False
        self._was_collided = False
        self._repair_count = 0
        self._clarification_count = 0
        self._turn_count = 0
        self._last_speech_end_ms: int | None = None

    def observe(self, event: ConversationEvent) -> None:
        if event.kind == "user_speech_start":
            if not self._is_user_speaking:
                self._turn_count += 1
            self._is_user_speaking = True
        elif event.kind == "user_speech_end":
            self._is_user_speaking = False
            self._last_speech_end_ms = event.timestamp_ms
        elif event.kind == "agent_output_queued":
            self._queued_output = True
        elif event.kind == "agent_speech_start":
            self._is_agent_speaking = True
            self._queued_output = False
        elif event.kind == "agent_speech_end":
            self._is_agent_speaking = False
            self._last_speech_end_ms = event.timestamp_ms
        elif event.kind == "agent_speech_yielded":
            self._is_agent_speaking = False
            self._has_deferred_output = True
            self._last_speech_end_ms = event.timestamp_ms
        elif event.kind == "overlap_detected":
            self._was_collided = True
            self._was_interrupted = True
            if self._is_agent_speaking or self._queued_output:
                self._has_deferred_output = True
        elif event.kind == "user_correction":
            self._needs_clarification = True
        elif event.kind == "repair_issued":
            self._repair_count += 1
            self._was_collided = False
            self._was_interrupted = False
            self._has_deferred_output = False
        elif event.kind == "clarification_issued":
            self._clarification_count += 1
            self._needs_clarification = False
        else:
            raise ValueError(f"unknown conversation event: {event.kind}")

    def get_conversation_features(self, *, now_ms: int) -> ConversationFeatures:
        if self._is_user_speaking or self._is_agent_speaking or self._last_speech_end_ms is None:
            silence_ms = 0
        else:
            silence_ms = max(0, int(now_ms) - self._last_speech_end_ms)
        return ConversationFeatures(
            is_user_speaking=self._is_user_speaking,
            is_agent_speaking=self._is_agent_speaking,
            queued_output=self._queued_output,
            has_deferred_output=self._has_deferred_output,
            needs_clarification=self._needs_clarification,
            was_interrupted=self._was_interrupted,
            was_collided=self._was_collided,
            silence_ms=silence_ms,
            repair_count=self._repair_count,
            clarification_count=self._clarification_count,
            turn_count=self._turn_count,
        )
