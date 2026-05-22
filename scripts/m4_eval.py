#!/usr/bin/env python3
"""M4 evaluation harness for distilled vs PPO policies.

Evaluates canonical + expanded scenario suites and reports per-case outcomes,
accuracy, and shaped reward totals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from m2_guarded_policy import GuardedDistilledPolicy
from m2_train import DistilledSpeakingPolicy
from m3_scenario_benchmark import SCENARIO_SUITE
from m4_rollout_env import RewardedScenarioEnv

ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]
from m4_train_ppo import ActorCritic


def load_policy(kind: str, checkpoint: str, device: torch.device):
    kind = kind.lower()
    if kind == "distilled":
        model = DistilledSpeakingPolicy(hidden_dim=64, dropout=0.1)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model
    if kind == "guarded":
        model = DistilledSpeakingPolicy(hidden_dim=64, dropout=0.1)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return GuardedDistilledPolicy(model)
    if kind == "ppo":
        model = ActorCritic(hidden_dim=64, dropout=0.1)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["policy_state"])
        model.to(device).eval()
        return model
    raise ValueError(f"Unknown policy kind: {kind}")


def predict_action(policy: Any, features: Dict[str, Any], device: torch.device) -> str:
    if hasattr(policy, "predict"):
        pred = policy.predict(features)
        if isinstance(pred, tuple):
            return pred[0]
        return pred
    # PPO actor-critic
    from m2_train import encode_features

    x = torch.tensor(encode_features(features), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy(x)
        return ACTION_NAMES[int(torch.argmax(logits, dim=-1).item())]


def evaluate_policy(policy: Any, device: torch.device) -> Dict[str, Any]:
    env = RewardedScenarioEnv(SCENARIO_SUITE)
    rows = []
    correct = 0
    total_reward = 0.0
    by_category = {}
    for i, scenario in enumerate(SCENARIO_SUITE):
        obs = env.reset(i)
        action = predict_action(policy, obs, device)
        _, reward, _, info = env.step(action)
        ok = action == info["expected_action"]
        correct += int(ok)
        total_reward += reward
        cat = info["category"]
        by_category.setdefault(cat, {"correct": 0, "total": 0})
        by_category[cat]["correct"] += int(ok)
        by_category[cat]["total"] += 1
        rows.append(
            {
                "name": info["scenario_name"],
                "category": cat,
                "expected": info["expected_action"],
                "predicted": action,
                "reward": reward,
                "correct": ok,
                "reason": info["reason"],
            }
        )

    return {
        "accuracy": correct / len(SCENARIO_SUITE),
        "total_reward": total_reward,
        "category_accuracy": {k: v["correct"] / v["total"] for k, v in by_category.items()},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate M4 policies")
    parser.add_argument("--policy-kind", choices=["distilled", "guarded", "ppo"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Distilled checkpoint or PPO checkpoint")
    parser.add_argument("--output", default=None, help="Optional JSON output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy_kind, args.checkpoint, device)
    report = {
        "policy_kind": args.policy_kind,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "scenario_report": evaluate_policy(policy, device),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
