"""M8 — Shaped reward function for speaking-policy evaluation.

Extends M4 reward with conversational quality metrics:
- repair success (user continues after repair)
- clarification efficiency (user clarifies and conversation continues)
- smooth turn transitions (natural handoff)
- missed repairs, unnecessary clarifies, conversation stalls

Usage:
    config = M8RewardConfig()
    signal = compute_m8_reward(config, action, features, next_features)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class M8RewardConfig:
    """Reward weights for M8 conversational quality signals.

    These are in addition to the M4 base rewards (speak_on_silence,
    wait_while_user_speaks, etc.) from RewardConfig.
    """
    # Positive signals
    successful_repair: float = 1.5
    successful_clarify: float = 1.0
    smooth_turn_transition: float = 0.5
    resume_after_yield: float = 0.8

    # Negative signals
    missed_repair: float = -1.5
    unnecessary_clarify: float = -0.5
    stall_per_second: float = -0.2       # per second of dead air beyond threshold
    stall_threshold_s: float = 3.0       # how much dead air before stall penalty
    self_interrupt: float = -1.0         # agent interrupted itself (yield while no user speech)
    long_monologue: float = -0.3         # per second beyond max_monologue_duration
    max_monologue_duration: float = 10.0  # seconds before monologue penalty


@dataclass(frozen=True)
class M8RewardSignal:
    """A single reward signal from the M8 reward function."""
    reward: float
    reason: str
    components: Dict[str, float]


def compute_m8_reward(
    config: M8RewardConfig,
    action: str,
    features: Dict[str, Any],
    next_features: Dict[str, Any],
) -> M8RewardSignal:
    """Compute M8 shaped reward for a single (action, features) step.

    Parameters
    ----------
    config : M8RewardConfig
        Reward weights.
    action : str
        The action taken by the policy (one of ACTION_NAMES).
    features : dict
        Conversation features BEFORE the action was taken.
    next_features : dict
        Conversation features AFTER the action was taken.

    Returns
    -------
    M8RewardSignal
        The reward, reason string, and component breakdown.
    """
    components: Dict[str, float] = {}

    user_was_speaking = features.get("is_user_speaking", False)
    user_is_speaking = next_features.get("is_user_speaking", False)
    was_agent_speaking = features.get("is_agent_speaking", False)
    is_agent_speaking = next_features.get("is_agent_speaking", False)
    was_deferred = features.get("is_agent_deferred", False)
    is_deferred = next_features.get("is_agent_deferred", False)
    was_repair = features.get("just_collided", False)
    was_interrupted = features.get("just_interrupted", False)
    just_resumed = next_features.get("just_resumed", False)
    seconds_since_user = next_features.get("seconds_since_last_user_speech", 999.0)
    agent_speak_duration = next_features.get("seconds_agent_has_been_speaking", 0.0)
    next_state = next_features.get("conversation_state", "idle")

    # 1. Successful repair: repair action → user speaks within 2s and agent is listening
    if action == "repair" or was_repair:
        if user_is_speaking and not is_agent_speaking:
            components["successful_repair"] = config.successful_repair
        elif seconds_since_user > 5.0 and not user_is_speaking:
            # Repair happened but user didn't respond — may be missed repair
            pass  # no positive signal

    # 2. Successful clarify: clarify action → user responds
    if action == "clarify":
        if user_is_speaking and not is_agent_speaking:
            components["successful_clarify"] = config.successful_clarify
        elif seconds_since_user > 4.0:
            components["unnecessary_clarify"] = config.unnecessary_clarify

    # 3. Smooth turn transition: agent finishes → user starts within 1.5s
    if (was_agent_speaking and not is_agent_speaking and
            user_is_speaking and seconds_since_user < 1.5):
        components["smooth_turn_transition"] = config.smooth_turn_transition

    # 4. Resume after yield: agent was deferred, now resumes
    if just_resumed:
        components["resume_after_yield"] = config.resume_after_yield

    # 5. Yield detection
    if action == "yield":
        if was_interrupted or user_was_speaking:
            # Correct yield — handled by M4 base reward
            pass
        elif not user_was_speaking and was_agent_speaking:
            components["self_interrupt"] = config.self_interrupt

    # 6. Conversation stall: agent expected to speak but dead air
    if (action == "wait" and not user_is_speaking and not is_agent_speaking and
            features.get("has_queued_output", False)):
        silence = seconds_since_user
        if silence > config.stall_threshold_s:
            stall_time = min(silence - config.stall_threshold_s, 5.0)  # cap at 5s per step
            components["stall"] = config.stall_per_second * stall_time

    # 7. Long monologue: agent speaking for too long
    if is_agent_speaking and agent_speak_duration > config.max_monologue_duration:
        overtime = agent_speak_duration - config.max_monologue_duration
        components["long_monologue"] = config.long_monologue * overtime

    # Compute total
    total_reward = sum(components.values())

    # Build reason string
    if components:
        reason_parts = [f"{k}={v:+.2f}" for k, v in components.items()]
        reason = "; ".join(reason_parts)
    else:
        reason = "no_m8_signal"

    return M8RewardSignal(
        reward=total_reward,
        reason=reason,
        components=components,
    )
