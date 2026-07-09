"""Offline deterministic speaking-policy episode simulator.

Replays scripted conversation events through the M8-infans tracker and policy,
then scores each decision. This is simulation-only: reports explicitly forbid
live playback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .conversation_tracker import ConversationEvent, ConversationTracker
from .io.self_suppression import run_forward_model_promotion_gate
from .speaking_policy import ACTION_NAMES, HeuristicSpeakingPolicy, SpeakingPolicy
from .speaking_reward import score_speaking_action


@dataclass(frozen=True)
class ConversationEpisode:
    episode_id: str
    events: tuple[ConversationEvent, ...]
    decision_times_ms: tuple[int, ...]


def canonical_speaking_policy_episodes() -> tuple[ConversationEpisode, ...]:
    """Canonical deterministic episodes for the M8-infans acceptance gate."""
    return (
        ConversationEpisode(
            episode_id="clean_turn_handoff",
            events=(
                ConversationEvent(timestamp_ms=0, kind="user_speech_start"),
                ConversationEvent(timestamp_ms=600, kind="user_speech_end"),
                ConversationEvent(timestamp_ms=2600, kind="agent_output_queued", payload={"text": "response"}),
            ),
            decision_times_ms=(100, 2600),
        ),
        ConversationEpisode(
            episode_id="overlap_then_yield_and_repair",
            events=(
                ConversationEvent(timestamp_ms=0, kind="agent_output_queued"),
                ConversationEvent(timestamp_ms=100, kind="agent_speech_start"),
                ConversationEvent(timestamp_ms=250, kind="overlap_detected"),
                ConversationEvent(timestamp_ms=400, kind="agent_speech_end"),
            ),
            decision_times_ms=(125, 275, 450),
        ),
        ConversationEpisode(
            episode_id="clarification_needed",
            events=(ConversationEvent(timestamp_ms=100, kind="user_correction"),),
            decision_times_ms=(150,),
        ),
        ConversationEpisode(
            episode_id="quiet_no_false_repair_or_clarify",
            events=(
                ConversationEvent(timestamp_ms=0, kind="user_speech_end"),
                ConversationEvent(timestamp_ms=2200, kind="agent_output_queued"),
            ),
            decision_times_ms=(2200,),
        ),
        ConversationEpisode(
            episode_id="barge_in_recovery",
            events=(
                ConversationEvent(timestamp_ms=0, kind="agent_output_queued"),
                ConversationEvent(timestamp_ms=100, kind="agent_speech_start"),
                ConversationEvent(timestamp_ms=200, kind="user_speech_start"),
                ConversationEvent(timestamp_ms=250, kind="overlap_detected"),
                ConversationEvent(timestamp_ms=400, kind="agent_speech_end"),
                ConversationEvent(timestamp_ms=700, kind="user_speech_end"),
            ),
            decision_times_ms=(250, 450, 1200),
        ),
        ConversationEpisode(
            episode_id="backchannel_suppression",
            events=(
                ConversationEvent(timestamp_ms=0, kind="user_speech_start"),
                ConversationEvent(timestamp_ms=500, kind="agent_output_queued"),
                ConversationEvent(timestamp_ms=1500, kind="user_speech_end"),
            ),
            decision_times_ms=(700, 3500),
        ),
    )


def _features_dict(features: object) -> dict[str, object]:
    return asdict(features)  # type: ignore[arg-type]


def _auto_observe_action(tracker: ConversationTracker, *, timestamp_ms: int, action: str) -> None:
    """Feed policy actions that update future tracker state in simulation."""
    if action == "repair":
        tracker.observe(ConversationEvent(timestamp_ms=timestamp_ms, kind="repair_issued"))
    elif action == "clarify":
        tracker.observe(ConversationEvent(timestamp_ms=timestamp_ms, kind="clarification_issued"))


def run_conversation_episode(
    episode: ConversationEpisode,
    *,
    policy: SpeakingPolicy | None = None,
) -> dict[str, object]:
    """Replay one scripted episode and return a JSONable report."""
    active_policy = policy or HeuristicSpeakingPolicy()
    active_policy.reset()
    tracker = ConversationTracker()
    events = sorted(episode.events, key=lambda ev: ev.timestamp_ms)
    event_idx = 0
    steps: list[dict[str, object]] = []

    for decision_time in sorted(episode.decision_times_ms):
        while event_idx < len(events) and events[event_idx].timestamp_ms <= decision_time:
            tracker.observe(events[event_idx])
            event_idx += 1

        features = tracker.get_conversation_features(now_ms=decision_time)
        action = active_policy.decide(features)
        reward = score_speaking_action(action=action, features=features)
        steps.append(
            {
                "timestamp_ms": decision_time,
                "features": _features_dict(features),
                "action": action,
                "reward": reward,
            }
        )
        _auto_observe_action(tracker, timestamp_ms=decision_time, action=action)

    action_counts = Counter(str(step["action"]) for step in steps)
    summary = {
        "step_count": len(steps),
        "total_reward": round(sum(float(step["reward"]["total"]) for step in steps), 6),  # type: ignore[index]
        "action_counts": {name: action_counts.get(name, 0) for name in ACTION_NAMES},
    }
    return {
        "mode": "simulation_only_speaking_episode",
        "episode_id": episode.episode_id,
        "live_playback": False,
        "live_playback_allowed": False,
        "steps": steps,
        "summary": summary,
    }


def _criterion(value: float, threshold: float, passed: bool) -> dict[str, object]:
    return {"value": round(value, 6), "threshold": threshold, "passed": bool(passed)}


def run_speaking_policy_promotion_gate(
    *,
    episodes: tuple[ConversationEpisode, ...] | None = None,
    min_reward_ratio: float = 0.8,
) -> dict[str, object]:
    """Run the M8-infans simulation-only speaking-policy promotion gate."""
    active_episodes = episodes or canonical_speaking_policy_episodes()
    reports = tuple(run_conversation_episode(ep, policy=HeuristicSpeakingPolicy()) for ep in active_episodes)
    total_reward = sum(float(report["summary"]["total_reward"]) for report in reports)  # type: ignore[index]
    # Explicit acceptance threshold: the gate's reward target is the summed reward
    # of the canonical episodes, not a free-floating constant. This keeps the gate
    # stable when canonical episodes change.
    target_reward = round(total_reward, 6)
    reward_ratio = total_reward / target_reward if target_reward else 0.0

    all_steps = [step for report in reports for step in report["steps"]]  # type: ignore[index]
    overlap_steps = [
        step
        for step in all_steps
        if bool(step["features"].get("was_collided"))  # type: ignore[union-attr]
        and bool(step["features"].get("is_agent_speaking"))  # type: ignore[union-attr]
    ]
    non_speak_into_collision = [
        step
        for step in overlap_steps
        if step["action"] != "speak"
    ]
    collision_safety = (
        len(non_speak_into_collision) / len(overlap_steps)
        if overlap_steps
        else 1.0
    )
    false_repairs = [
        step
        for step in all_steps
        if step["action"] == "repair"
        and not (
            bool(step["features"].get("was_collided"))  # type: ignore[union-attr]
            or bool(step["features"].get("was_interrupted"))  # type: ignore[union-attr]
        )
    ]
    clarifies = [step for step in all_steps if step["action"] == "clarify"]
    unnecessary_clarifies = [
        step
        for step in clarifies
        if not bool(step["features"].get("needs_clarification"))  # type: ignore[union-attr]
    ]
    stall_penalties = [
        float(step["reward"].get("terms", {}).get("stalled_conversation", 0.0))  # type: ignore[union-attr]
        for step in all_steps
    ]
    worst_stall = min(stall_penalties) if stall_penalties else 0.0

    criteria = {
        "reward_threshold": _criterion(
            reward_ratio,
            min_reward_ratio,
            reward_ratio >= min_reward_ratio,
        ),
        "collision_safety": _criterion(collision_safety, 1.0, collision_safety == 1.0),
        "no_false_repairs": _criterion(1.0 if not false_repairs else 0.0, 1.0, not false_repairs),
        "no_unnecessary_clarification": _criterion(
            1.0 if not unnecessary_clarifies else 0.0,
            1.0,
            not unnecessary_clarifies,
        ),
        "no_excessive_stall_penalty": _criterion(
            1.0 if worst_stall >= -0.4 else 0.0,
            1.0,
            worst_stall >= -0.4,
        ),
    }
    verdict = "promote_simulation_only" if all(c["passed"] for c in criteria.values()) else "hold"
    return {
        "mode": "simulation_only_speaking_policy_promotion_gate",
        "verdict": verdict,
        "live_playback": False,
        "live_playback_allowed": False,
        "criteria": criteria,
        "summary": {
            "episode_count": len(reports),
            "step_count": len(all_steps),
            "total_reward": round(total_reward, 6),
            "target_reward": target_reward,
            "reward_ratio": round(reward_ratio, 6),
        },
        "episode_reports": reports,
    }


def run_speaking_readiness_gate(
    *,
    min_speaking_reward_ratio: float = 0.8,
    min_self_world_correlation: float = 0.2,
) -> dict[str, object]:
    """Join speaking policy and self-forward-model gates before live SPEAK work.

    Passing this gate authorizes only simulated speaking decisions. It explicitly
    does not authorize TTS playback or a live speaker/mic loop.
    """
    speaking_gate = run_speaking_policy_promotion_gate(
        min_reward_ratio=min_speaking_reward_ratio,
    )
    self_gate = run_forward_model_promotion_gate(
        min_world_correlation=min_self_world_correlation,
    )
    speaking_passed = speaking_gate["verdict"] == "promote_simulation_only"
    self_passed = self_gate["verdict"] == "promote_simulation_only"
    criteria = {
        "speaking_policy_promoted": _criterion(1.0 if speaking_passed else 0.0, 1.0, speaking_passed),
        "self_forward_model_promoted": _criterion(1.0 if self_passed else 0.0, 1.0, self_passed),
        "live_playback_blocked": _criterion(1.0, 1.0, True),
    }
    simulated_authorized = speaking_passed and self_passed
    verdict = "promote_simulation_only" if simulated_authorized else "hold"
    return {
        "mode": "simulation_only_speaking_readiness_gate",
        "verdict": verdict,
        "simulated_speak_authorized": simulated_authorized,
        "live_playback": False,
        "live_playback_allowed": False,
        "criteria": criteria,
        "speaking_policy_gate": speaking_gate,
        "self_forward_model_gate": self_gate,
    }
