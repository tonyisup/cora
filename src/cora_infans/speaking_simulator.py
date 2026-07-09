"""Offline deterministic speaking-policy episode simulator.

Replays scripted conversation events through the M8-infans tracker and policy,
then scores each decision. This is simulation-only: reports explicitly forbid
live playback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .conversation_tracker import ConversationEvent, ConversationTracker
from .speaking_policy import ACTION_NAMES, HeuristicSpeakingPolicy, SpeakingPolicy
from .speaking_reward import score_speaking_action


@dataclass(frozen=True)
class ConversationEpisode:
    episode_id: str
    events: tuple[ConversationEvent, ...]
    decision_times_ms: tuple[int, ...]


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
