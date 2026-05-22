#!/usr/bin/env python3
"""M4 multi-step turn-taking environment scaffold.

This environment turns the previous one-step scenario scoring into a short
sequence problem. Each episode lasts multiple steps and the next observation
changes based on the previous action, so PPO can see delayed consequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from m4_reward import RewardSignal, reward_from_transition


ACTIONS = ["speak", "wait", "clarify", "repair", "yield"]


@dataclass(frozen=True)
class MultiStepTransition:
    episode_id: int
    step_idx: int
    observation: Dict[str, Any]
    action: str
    reward: float
    next_observation: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


@dataclass(frozen=True)
class EpisodeStepTemplate:
    name: str
    base_obs: Dict[str, Any]
    transition: Callable[[Dict[str, Any], str, int], Dict[str, Any]]
    terminal_after: int = 3


def _obs(**kwargs: Any) -> Dict[str, Any]:
    base = {
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 0.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "queue_depth": 0,
    }
    base.update(kwargs)
    return base


def _transition_idle_no_queue(obs: Dict[str, Any], action: str, step_idx: int) -> Dict[str, Any]:
    # Delayed consequence: the wrong action on step 0 makes step 1 harder.
    if step_idx == 0:
        if action == "wait":
            return _obs(**{**obs, "silence_duration_ms": 1200.0, "queue_depth": 1, "has_queued_utterance": True})
        if action == "clarify":
            return _obs(state="repair", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)
        if action == "speak":
            return _obs(state="speak", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
    if step_idx == 1:
        if action == "speak":
            return _obs(state="speak", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=1)
        if action == "wait":
            return _obs(state="listen", attribution="silence", silence_duration_ms=2300.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)
        if action == "clarify":
            return _obs(state="clarify", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
    return _obs(state="listen", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)


def _transition_interrupt_chain(obs: Dict[str, Any], action: str, step_idx: int) -> Dict[str, Any]:
    # The action decides whether the episode cools down or becomes a repair loop.
    if step_idx == 0:
        if action == "yield":
            return _obs(state="listen", attribution="world", silence_duration_ms=250.0, has_queued_utterance=True, user_speaking=False, agent_speaking=False, queue_depth=1)
        if action == "wait":
            return _obs(state="speak", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
        return _obs(state="speak", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
    if step_idx == 1:
        if action == "yield":
            return _obs(state="listen", attribution="silence", silence_duration_ms=150.0, has_queued_utterance=True, user_speaking=False, agent_speaking=False, queue_depth=1)
        if action == "repair":
            return _obs(state="repair", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
        return _obs(state="speak", attribution="both", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
    return _obs(state="listen", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)


def _transition_repair_chain(obs: Dict[str, Any], action: str, step_idx: int) -> Dict[str, Any]:
    if step_idx == 0:
        if action == "repair":
            return _obs(state="repair", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
        return _obs(state="repair", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
    if step_idx == 1:
        if action == "wait":
            return _obs(state="listen", attribution="silence", silence_duration_ms=900.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)
        if action == "speak":
            return _obs(state="speak", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=True, user_speaking=False, agent_speaking=True, queue_depth=1)
        return _obs(state="repair", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
    return _obs(state="listen", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)


def _transition_silence_to_speak(obs: Dict[str, Any], action: str, step_idx: int) -> Dict[str, Any]:
    if step_idx == 0:
        if action == "speak":
            return _obs(state="speak", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=True, user_speaking=False, agent_speaking=True, queue_depth=1)
        if action == "wait":
            return _obs(state="listen", attribution="silence", silence_duration_ms=2200.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)
        return _obs(state="clarify", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
    if step_idx == 1:
        if action == "wait":
            return _obs(state="listen", attribution="silence", silence_duration_ms=2400.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)
        if action == "clarify":
            return _obs(state="clarify", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True, queue_depth=0)
        return _obs(state="speak", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True, queue_depth=0)
    return _obs(state="listen", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False, queue_depth=0)


EPISODE_TEMPLATES: List[EpisodeStepTemplate] = [
    EpisodeStepTemplate("idle_no_queue", _obs(state="listen", attribution="silence", silence_duration_ms=150.0, has_queued_utterance=False, user_speaking=False, agent_speaking=False), _transition_idle_no_queue),
    EpisodeStepTemplate("interrupt_chain", _obs(state="speak", attribution="world", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=True, agent_speaking=True), _transition_interrupt_chain),
    EpisodeStepTemplate("repair_chain", _obs(state="repair", attribution="silence", silence_duration_ms=0.0, has_queued_utterance=False, user_speaking=False, agent_speaking=True), _transition_repair_chain),
    EpisodeStepTemplate("silence_to_speak", _obs(state="listen", attribution="silence", silence_duration_ms=1800.0, has_queued_utterance=True, user_speaking=False, agent_speaking=False), _transition_silence_to_speak),
]


class MultiStepTurnEnv:
    def __init__(self, templates: List[EpisodeStepTemplate] | None = None, horizon: int = 3):
        self.templates = templates or EPISODE_TEMPLATES
        self.horizon = horizon
        self._episode_id = 0
        self._step_idx = 0
        self._template: EpisodeStepTemplate | None = None
        self._obs: Dict[str, Any] | None = None

    def reset(self, episode_id: int = 0) -> Dict[str, Any]:
        self._episode_id = episode_id
        self._step_idx = 0
        self._template = self.templates[episode_id % len(self.templates)]
        self._obs = dict(self._template.base_obs)
        return self._obs

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._template is None or self._obs is None:
            raise RuntimeError("Call reset() before step().")

        prev_obs = dict(self._obs)
        next_obs = self._template.transition(prev_obs, action, self._step_idx)
        signal = reward_from_transition(prev_obs, action, next_obs)
        done = self._step_idx + 1 >= self._template.terminal_after or self._step_idx + 1 >= self.horizon
        info = {
            "episode_id": self._episode_id,
            "episode_name": self._template.name,
            "step_idx": self._step_idx,
            "reward_reason": signal.reason,
            "reward_components": signal.components,
            "terminal_after": self._template.terminal_after,
        }
        self._obs = next_obs
        self._step_idx += 1
        return next_obs, signal.reward, done, info

    def rollout(self, policy) -> List[MultiStepTransition]:
        transitions: List[MultiStepTransition] = []
        for episode_id, template in enumerate(self.templates):
            obs = self.reset(episode_id)
            done = False
            local_step = 0
            while not done:
                action = policy.predict(obs) if hasattr(policy, "predict") else policy(obs)
                next_obs, reward, done, info = self.step(action)
                transitions.append(
                    MultiStepTransition(
                        episode_id=episode_id,
                        step_idx=local_step,
                        observation=obs,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        done=done,
                        info=info,
                    )
                )
                obs = next_obs
                local_step += 1
                if local_step > self.horizon + 1:
                    break
        return transitions


class HeuristicPolicy:
    def predict(self, features: Dict[str, Any]) -> str:
        state = str(features.get("state", "")).lower()
        attr = str(features.get("attribution", "")).lower()
        silence = float(features.get("silence_duration_ms", 0.0) or 0.0)
        has_queue = bool(features.get("has_queued_utterance", False))
        user_speaking = bool(features.get("user_speaking", False))

        if state == "repair":
            return "repair"
        if state == "speak" and (user_speaking or attr in {"world", "both"}):
            return "yield"
        if state == "listen" and not has_queue and silence < 1000.0:
            return "wait"
        if state == "listen" and not has_queue and silence >= 2000.0:
            return "clarify"
        if state == "listen" and has_queue:
            return "speak"
        if user_speaking:
            return "wait"
        return "wait"


if __name__ == "__main__":
    env = MultiStepTurnEnv()
    policy = HeuristicPolicy()
    transitions = env.rollout(policy)
    print(f"episodes={len(env.templates)} transitions={len(transitions)}")
    for t in transitions:
        print(
            f"ep={t.episode_id} step={t.step_idx} {t.action} -> reward={t.reward:+.2f} done={t.done} episode={t.info['episode_name']} reason={t.info['reward_reason']}"
        )
