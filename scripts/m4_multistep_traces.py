#!/usr/bin/env python3
"""M4 multi-step trace logger.

Writes JSONL traces from the multi-step turn environment so reward causality can
be inspected and later consumed by a PPO trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from m4_multistep_env import MultiStepTurnEnv, HeuristicPolicy


def collect_traces(episodes: int) -> List[Dict[str, Any]]:
    env = MultiStepTurnEnv()
    policy = HeuristicPolicy()
    rows: List[Dict[str, Any]] = []

    for episode_id in range(episodes):
        obs = env.reset(episode_id)
        done = False
        step_idx = 0
        while not done:
            action = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_name": info["episode_name"],
                    "step_idx": step_idx,
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_obs,
                    "done": done,
                    "reason": info["reward_reason"],
                    "components": info["reward_components"],
                }
            )
            obs = next_obs
            step_idx += 1
            if step_idx > env.horizon + 1:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect M4 multi-step traces")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", default="results/m4_multistep_traces.jsonl")
    args = parser.parse_args()

    rows = collect_traces(args.episodes)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"Wrote {len(rows)} transitions -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
