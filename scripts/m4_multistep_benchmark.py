#!/usr/bin/env python3
"""M4 multi-step benchmark.

Evaluates policies on the new multi-step turn-taking environment and reports
sequence-level return plus step-level correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from m2_train import DistilledSpeakingPolicy
from m4_multistep_env import HeuristicPolicy, MultiStepTurnEnv
from m4_train_ppo import ActorCritic

ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]


def load_policy(kind: str, checkpoint: str | None, device: torch.device):
    kind = kind.lower()
    if kind == "heuristic":
        return HeuristicPolicy()
    if kind == "distilled":
        model = DistilledSpeakingPolicy(hidden_dim=64, dropout=0.1)
        if not checkpoint:
            raise ValueError("distilled policy requires --checkpoint")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model
    if kind == "ppo":
        model = ActorCritic(hidden_dim=64, dropout=0.1)
        if not checkpoint:
            raise ValueError("ppo policy requires --checkpoint")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["policy_state"])
        model.to(device).eval()
        return model
    raise ValueError(f"Unknown policy kind: {kind}")


def predict_action(policy: Any, obs: Dict[str, Any], device: torch.device) -> str:
    if hasattr(policy, "predict"):
        pred = policy.predict(obs)
        return pred[0] if isinstance(pred, tuple) else pred
    from m2_train import encode_features

    x = torch.tensor(encode_features(obs), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy(x)
        return ACTION_NAMES[int(torch.argmax(logits, dim=-1).item())]


def evaluate(policy: Any, episodes: int, device: torch.device) -> Dict[str, Any]:
    env = MultiStepTurnEnv()
    rows: List[Dict[str, Any]] = []
    total_return = 0.0
    correct = 0
    total_steps = 0
    episode_returns = []

    for episode_id in range(episodes):
        obs = env.reset(episode_id)
        done = False
        ep_return = 0.0
        step_idx = 0
        while not done:
            action = predict_action(policy, obs, device)
            next_obs, reward, done, info = env.step(action)
            ok = False
            # Multi-step benchmark measures whether the policy matches the current step's ideal action
            # by using the same shaped reward as the learning loop. The sequence return is the main metric.
            if reward >= 0:
                ok = True
            correct += int(ok)
            total_steps += 1
            ep_return += reward
            rows.append(
                {
                    "episode_id": episode_id,
                    "episode_name": info["episode_name"],
                    "step_idx": step_idx,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "reason": info["reward_reason"],
                    "components": info["reward_components"],
                    "observation": obs,
                    "next_observation": next_obs,
                }
            )
            obs = next_obs
            step_idx += 1
        episode_returns.append(ep_return)
        total_return += ep_return

    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "step_positive_rate": correct / max(total_steps, 1),
        "mean_episode_return": total_return / max(episodes, 1),
        "episode_returns": episode_returns,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 multi-step benchmark")
    parser.add_argument("--policy-kind", choices=["heuristic", "distilled", "ppo"], default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.policy_kind, args.checkpoint, device)
    report = {
        "policy_kind": args.policy_kind,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "benchmark": evaluate(policy, args.episodes, device),
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
