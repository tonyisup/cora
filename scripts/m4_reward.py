#!/usr/bin/env python3
"""M4 shaped reward utilities for speaking-policy RL.

This module defines a small, explicit reward function for turn-taking.
It is intentionally conservative: the reward is designed to prefer the
canonical safe behavior and only reward ambiguous behavior when the
scenario context makes it clearly appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class RewardConfig:
    speak_on_silence: float = 1.0
    wait_while_user_speaks: float = 0.35
    wait_in_idle_no_queue: float = 0.25
    yield_on_interrupt: float = 2.0
    yield_on_collision: float = 2.0
    clarify_after_long_silence: float = 0.75
    repair_in_repair_state: float = 1.5
    premature_clarify: float = -0.9
    premature_speak: float = -1.4
    missed_interrupt: float = -2.0
    bad_yield: float = -1.0
    bad_repair: float = -0.7


@dataclass(frozen=True)
class RewardSignal:
    reward: float
    reason: str
    components: Dict[str, float]


def _get_bool(features: Dict[str, Any], key: str) -> bool:
    return bool(features.get(key, False))


def _get_float(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except (TypeError, ValueError):
        return default


def _state(features: Dict[str, Any]) -> str:
    return str(features.get("state", "")).strip().lower()


def _attr(features: Dict[str, Any]) -> str:
    return str(features.get("attribution", "")).strip().lower()


def compute_reward(
    features: Dict[str, Any],
    action: str,
    *,
    cfg: RewardConfig | None = None,
) -> RewardSignal:
    """Score a single policy decision.

    The reward is based on the current observation and the chosen action.
    It is designed for offline RL/PPO rollouts in the speaking-policy simulator.
    """

    cfg = cfg or RewardConfig()
    state = _state(features)
    attr = _attr(features)
    silence_ms = _get_float(features, "silence_duration_ms")
    has_queue = _get_bool(features, "has_queued_utterance")
    user_speaking = _get_bool(features, "user_speaking")
    agent_speaking = _get_bool(features, "agent_speaking")

    components: Dict[str, float] = {}
    action = action.strip().lower()

    if action == "speak":
        if state == "listen" and not user_speaking and has_queue:
            components["speak_on_silence"] = cfg.speak_on_silence
        elif state == "speak" and attr in {"world", "both"}:
            components["premature_speak"] = cfg.premature_speak
        elif user_speaking:
            components["premature_speak"] = cfg.premature_speak
        else:
            components["premature_speak"] = cfg.premature_speak * 0.5

    elif action == "wait":
        if user_speaking:
            components["wait_while_user_speaks"] = cfg.wait_while_user_speaks
        elif state == "listen" and not has_queue:
            # This is the exact ambiguous boundary that M3 exposed.
            components["wait_in_idle_no_queue"] = cfg.wait_in_idle_no_queue
        elif state == "listen" and has_queue:
            components["wait_with_queue"] = 0.05
        else:
            components["wait_generic"] = 0.1

    elif action == "yield":
        if state == "speak" and (user_speaking or attr in {"world", "both"}):
            components["yield_on_interrupt_or_collision"] = cfg.yield_on_interrupt if attr != "both" else cfg.yield_on_collision
        else:
            components["bad_yield"] = cfg.bad_yield

    elif action == "clarify":
        if state == "listen" and not user_speaking and silence_ms >= 2000.0:
            components["clarify_after_long_silence"] = cfg.clarify_after_long_silence
        elif state == "listen" and not user_speaking and silence_ms < 1000.0 and not has_queue:
            components["premature_clarify"] = cfg.premature_clarify
        else:
            components["clarify_generic"] = cfg.premature_clarify * 0.5

    elif action == "repair":
        if state == "repair":
            components["repair_in_repair_state"] = cfg.repair_in_repair_state
        else:
            components["bad_repair"] = cfg.bad_repair

    else:
        components["unknown_action"] = -1.0

    reward = sum(components.values())

    if state == "speak" and user_speaking and action != "yield":
        components["missed_interrupt"] = cfg.missed_interrupt
        reward += cfg.missed_interrupt

    if state == "listen" and not user_speaking and not has_queue and silence_ms < 1000.0 and action == "clarify":
        # Re-emphasize the exact ambiguity boundary with a small extra penalty.
        components["idle_no_queue_clarify_penalty"] = cfg.premature_clarify * 0.5
        reward += cfg.premature_clarify * 0.5

    reason = ", ".join(f"{k}={v:+.2f}" for k, v in components.items()) or "neutral"
    return RewardSignal(reward=reward, reason=reason, components=components)


def reward_from_transition(
    prev_features: Dict[str, Any],
    action: str,
    next_features: Dict[str, Any],
    *,
    cfg: RewardConfig | None = None,
) -> RewardSignal:
    """Reward a transition.

    For now this delegates to the current observation and adds a tiny
    continuity bonus for settling into a quieter state after yielding.
    """

    signal = compute_reward(prev_features, action, cfg=cfg)
    cfg = cfg or RewardConfig()

    prev_user = _get_bool(prev_features, "user_speaking")
    next_user = _get_bool(next_features, "user_speaking")
    prev_agent = _get_bool(prev_features, "agent_speaking")
    next_agent = _get_bool(next_features, "agent_speaking")

    continuity = 0.0
    if action.strip().lower() == "yield" and prev_agent and not next_agent and next_user:
        continuity += 0.1
    if action.strip().lower() == "wait" and prev_user and next_user:
        continuity += 0.05

    if continuity:
        signal = RewardSignal(
            reward=signal.reward + continuity,
            reason=signal.reason + f", continuity={continuity:+.2f}",
            components={**signal.components, "continuity": continuity},
        )
    return signal


if __name__ == "__main__":
    # Tiny smoke test when run directly.
    samples = [
        ("idle_wait", {"state": "listen", "attribution": "silence", "silence_duration_ms": 150.0, "has_queued_utterance": False, "user_speaking": False, "agent_speaking": False}, "wait"),
        ("idle_clarify", {"state": "listen", "attribution": "silence", "silence_duration_ms": 150.0, "has_queued_utterance": False, "user_speaking": False, "agent_speaking": False}, "clarify"),
        ("interrupt_yield", {"state": "speak", "attribution": "world", "silence_duration_ms": 0.0, "has_queued_utterance": False, "user_speaking": True, "agent_speaking": True}, "yield"),
    ]
    for name, features, action in samples:
        sig = compute_reward(features, action)
        print(f"{name}: {action} -> {sig.reward:+.2f} ({sig.reason})")
