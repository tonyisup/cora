"""Simulation-only speaking policy scaffold.

Adapted from og-cora M8's policy layer, but deliberately safe for cora-infans:
this module chooses *would-be* speaking actions from conversation features. It
performs no TTS, no live playback, and no microphone control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

SpeakingAction = Literal["speak", "wait", "yield", "repair", "clarify"]
ACTION_NAMES: tuple[SpeakingAction, ...] = ("speak", "wait", "yield", "repair", "clarify")


@dataclass(frozen=True)
class ConversationFeatures:
    """Policy-facing conversation state.

    This is intentionally a small, explicit feature vector so future tracker /
    simulator work can feed the same deterministic policy without live audio.
    """

    is_user_speaking: bool = False
    is_agent_speaking: bool = False
    queued_output: bool = False
    has_deferred_output: bool = False
    needs_clarification: bool = False
    was_interrupted: bool = False
    was_collided: bool = False
    silence_ms: int = 0
    repair_count: int = 0
    clarification_count: int = 0
    turn_count: int = 0


class SpeakingPolicy(Protocol):
    def decide(self, features: ConversationFeatures) -> SpeakingAction:
        """Choose the next speaking action from conversation features."""
        ...

    def reset(self) -> None:
        """Reset policy-internal state, if any."""
        ...


class HeuristicSpeakingPolicy:
    """Deterministic M8-style speaking policy.

    Rule priority mirrors og-cora's scoped M8 policy: protect user speech first,
    yield on collision, repair interrupted/deferred output, clarify when needed,
    speak only after enough silence with queued output, otherwise wait.
    """

    def __init__(self, *, min_silence_before_speak_ms: int = 2000) -> None:
        if min_silence_before_speak_ms < 0:
            raise ValueError("min_silence_before_speak_ms must be non-negative")
        self.min_silence_before_speak_ms = min_silence_before_speak_ms

    def decide(self, features: ConversationFeatures) -> SpeakingAction:
        if features.is_user_speaking:
            return "wait"
        if features.is_agent_speaking and features.was_collided:
            return "yield"
        if features.was_interrupted and features.has_deferred_output:
            return "repair"
        if features.needs_clarification:
            return "clarify"
        if features.queued_output and features.silence_ms >= self.min_silence_before_speak_ms:
            return "speak"
        return "wait"

    def reset(self) -> None:
        return None
