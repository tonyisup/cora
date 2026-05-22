#!/usr/bin/env python3
"""M4 reward-trace collector.

Scores the expanded scenario bank under the shaped reward function and writes
an explicit JSONL trace of (scenario, action, reward, reason).
This gives PPO a stable reward reference and a human-auditable artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from m3_scenario_benchmark import SCENARIO_SUITE
from m4_reward import RewardConfig, compute_reward


ACTIONS = ["speak", "wait", "clarify", "repair", "yield"]


def collect_reward_traces(output_path: Path, *, cfg: RewardConfig | None = None) -> int:
    cfg = cfg or RewardConfig()
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_SUITE:
        features = scenario["features"]
        for action in ACTIONS:
            signal = compute_reward(features, action, cfg=cfg)
            rows.append(
                {
                    "scenario_name": scenario["name"],
                    "category": scenario.get("category", "canonical"),
                    "expected_action": scenario["expected"],
                    "action": action,
                    "reward": signal.reward,
                    "reason": signal.reason,
                    "components": signal.components,
                    "features": features,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect M4 shaped-reward traces")
    parser.add_argument("--output", default="results/m4_reward_traces.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    out = Path(args.output)
    n = collect_reward_traces(out)
    print(f"Wrote {n} reward rows -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
