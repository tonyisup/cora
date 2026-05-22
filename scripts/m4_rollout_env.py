#!/usr/bin/env python3
"""M4 shaped rollout environment scaffold.

This is a lightweight environment wrapper around the expanded scenario suite.
It is not the final live-audio environment; it exists to make the reward
loop concrete and give PPO a stable rollout API to target next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from m3_scenario_benchmark import SCENARIO_SUITE
from m4_reward import RewardSignal, compute_reward, reward_from_transition


ACTIONS = ["speak", "wait", "clarify", "repair", "yield"]


@dataclass
class RolloutStep:
    scenario_name: str
    category: str
    features: Dict[str, Any]
    action: str
    reward: float
    done: bool
    info: Dict[str, Any]


class RewardedScenarioEnv:
    """A tiny deterministic environment over the expanded scenario bank.

    The environment exposes one scenario per episode. Each `step(action)`
    returns the shaped reward for the chosen action and then terminates.
    This is intentionally simple: it provides a stable contract for M4
    before the project commits to a full simulator-backed PPO loop.
    """

    def __init__(self, scenarios: List[Dict[str, Any]] | None = None):
        self.scenarios = scenarios or SCENARIO_SUITE
        self._index = 0
        self._current: Dict[str, Any] | None = None
        self._last_signal: RewardSignal | None = None

    def reset(self, index: int = 0) -> Dict[str, Any]:
        self._index = index % len(self.scenarios)
        self._current = self.scenarios[self._index]
        self._last_signal = None
        return self._current["features"]

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("Call reset() before step().")

        features = self._current["features"]
        signal = compute_reward(features, action)
        self._last_signal = signal
        done = True
        info = {
            "scenario_name": self._current["name"],
            "category": self._current.get("category", "canonical"),
            "expected_action": self._current["expected"],
            "reason": signal.reason,
            "components": signal.components,
        }
        next_obs = features
        return next_obs, signal.reward, done, info

    def rollout(self, action_fn) -> List[RolloutStep]:
        steps: List[RolloutStep] = []
        for i, scenario in enumerate(self.scenarios):
            obs = self.reset(i)
            action = action_fn(obs)
            _, reward, done, info = self.step(action)
            steps.append(
                RolloutStep(
                    scenario_name=scenario["name"],
                    category=scenario.get("category", "canonical"),
                    features=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )
        return steps


class HeuristicPolicy:
    """A conservative reference policy for smoke testing the rollout env."""

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
        if state == "listen" and user_speaking:
            return "wait"
        if state == "listen" and not has_queue and silence >= 2000.0:
            return "clarify"
        if state == "listen" and has_queue:
            return "speak"
        return "wait"


def run_policy(policy, env: RewardedScenarioEnv | None = None) -> Dict[str, Any]:
    env = env or RewardedScenarioEnv()
    steps = env.rollout(lambda obs: policy.predict(obs))
    total_reward = sum(s.reward for s in steps)
    correct = sum(1 for s in steps if s.action == s.info["expected_action"])
    return {
        "episodes": len(steps),
        "accuracy": correct / len(steps) if steps else 0.0,
        "total_reward": total_reward,
        "steps": [
            {
                "scenario_name": s.scenario_name,
                "category": s.category,
                "action": s.action,
                "reward": s.reward,
                "expected_action": s.info["expected_action"],
                "reason": s.info["reason"],
            }
            for s in steps
        ],
    }


if __name__ == "__main__":
    report = run_policy(HeuristicPolicy())
    print(f"episodes={report['episodes']} accuracy={report['accuracy']:.3f} total_reward={report['total_reward']:+.2f}")
    for row in report["steps"]:
        print(f"- {row['scenario_name']}: {row['action']} vs {row['expected_action']} reward={row['reward']:+.2f}")
