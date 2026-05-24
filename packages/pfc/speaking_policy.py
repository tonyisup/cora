"""PFC — Speaking Policy interface and heuristic baseline for M8.

The SpeakingPolicy decides which conversational action to take given
the current conversation state. This is the decision layer that sits
above the reactive controller.

Architecture:
    - SpeakingPolicy (ABC): interface all policies must implement
    - HeuristicSpeakingPolicy: deterministic rule-based baseline
    - (future) DistilledSpeakingPolicy: MLP trained from heuristic demonstrations
    - (future) PPOSpeakingPolicy: RL-trained policy

Usage:
    policy = HeuristicSpeakingPolicy()
    action = policy.decide(features)  # returns one of ACTION_NAMES
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]

# Thresholds for the heuristic policy
SILENCE_BEFORE_SPEAK_S = 2.0       # speak after 2s of silence when output queued
SILENCE_BEFORE_CLARIFY_S = 4.0     # clarify after 4s of user silence (user may be confused)
MIN_SPEAK_DURATION_S = 0.3         # don't interrupt self too quickly
MAX_SPEAK_WITHOUT_USER_S = 8.0     # if agent has been speaking for 8s straight, yield


class SpeakingPolicy(ABC):
    """Interface for all speaking policies.

    Policies are stateless between calls — all state is carried in the
    conversation features dict produced by ConversationTracker.

    The policy's job is to map features → action string.
    """

    @abstractmethod
    def decide(self, features: Dict[str, Any]) -> str:
        """Decide the next conversational action.

        Parameters
        ----------
        features : dict
            Conversation features from ConversationTracker.get_features().

        Returns
        -------
        str
            One of: "speak", "wait", "clarify", "repair", "yield"
        """
        ...

    def reset(self) -> None:
        """Reset any internal state. Default: no-op."""
        pass


class HeuristicSpeakingPolicy(SpeakingPolicy):
    """Deterministic rule-based speaking policy.

    Encodes the baseline conversational behavior:
    - Wait while user is speaking
    - Speak when ready and no one else is talking
    - Yield immediately on interruption
    - Repair after collision
    - Clarify after long user silence when agent has deferred output

    This is the M8a baseline. It should be good enough to pass acceptance,
    and its demonstrations are used to train M8b distilled policy.
    """

    def __init__(
        self,
        silence_before_speak: float = SILENCE_BEFORE_SPEAK_S,
        silence_before_clarify: float = SILENCE_BEFORE_CLARIFY_S,
        min_speak_duration: float = MIN_SPEAK_DURATION_S,
        max_speak_without_user: float = MAX_SPEAK_WITHOUT_USER_S,
    ) -> None:
        self._silence_before_speak = silence_before_speak
        self._silence_before_clarify = silence_before_clarify
        self._min_speak_duration = min_speak_duration
        self._max_speak_without_user = max_speak_without_user

    def decide(self, features: Dict[str, Any]) -> str:
        is_user_speaking = features.get("is_user_speaking", False)
        is_agent_speaking = features.get("is_agent_speaking", False)
        is_agent_deferred = features.get("is_agent_deferred", False)
        has_queued_output = features.get("has_queued_output", False)
        just_interrupted = features.get("just_interrupted", False)
        just_collided = features.get("just_collided", False)
        seconds_since_user = features.get("seconds_since_last_user_speech", 999.0)
        seconds_since_agent = features.get("seconds_since_last_agent_speech", 999.0)
        agent_speak_duration = features.get("seconds_agent_has_been_speaking", 0.0)
        repair_count = features.get("repair_count", 0)
        conversation_state = features.get("conversation_state", "idle")

        # 1. Highest priority: yield on interruption
        if just_interrupted or (is_user_speaking and is_agent_speaking):
            return "yield"

        # 2. Repair after collision
        if just_collided or conversation_state == "repair":
            return "repair"

        # 3. If user is speaking, wait (even if we have queued output)
        if is_user_speaking:
            return "wait"

        # 4. If agent has deferred output (was interrupted), resume speaking
        if is_agent_deferred and not is_user_speaking:
            return "speak"

        # 5. If we have queued output and there's been enough silence, speak
        if has_queued_output:
            if seconds_since_user >= self._silence_before_speak:
                return "speak"
            return "wait"

        # 6. Agent is speaking — check if we've been talking too long
        if is_agent_speaking:
            if agent_speak_duration > self._max_speak_without_user:
                return "yield"
            return "speak"  # keep speaking

        # 7. User has been silent for a while and agent has deferred output → clarify
        if is_agent_deferred and seconds_since_user >= self._silence_before_clarify:
            return "clarify"

        # 8. Default: wait
        return "wait"
