#!/usr/bin/env python3
"""M8c — Evaluate PPO policy and compare against heuristic + distilled.

Runs the scripted episode suite through:
1. HeuristicSpeakingPolicy (baseline)
2. M8DistilledPolicy (M8b)
3. M8 PPO policy (M8c)

Reports action match rates and reward comparison.
M8c gate: PPO must beat distilled by >= 10% reward on expanded episodes.

Usage:
    .venv/bin/python scripts/m8_eval.py --ppo-checkpoint results/m8_ppo/best_policy.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.speaking_policy import HeuristicSpeakingPolicy, SpeakingPolicy, ACTION_NAMES
from m8_policy.model import M8DistilledPolicy, encode_m8_features
from m8_conversation_simulator import create_default_episodes, load_episodes, run_episode
from m8_reward import M8RewardConfig
from m8_train_ppo import M8ActorCritic, encode_obs


CANONICAL_EPISODES = {"speak_on_silence", "wait_while_user_speaks"}
MIN_PPO_IMPROVEMENT = 0.10  # PPO must beat distilled by >= 10% on expanded


class DistilledPolicyWrapper(SpeakingPolicy):
    def __init__(self, model: M8DistilledPolicy, device: torch.device):
        self._model = model
        self._device = device

    def decide(self, features: Dict[str, Any]) -> str:
        x = encode_m8_features(features).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            idx = logits.argmax(dim=-1).item()
        return ACTION_NAMES[idx]


class PPOPolicyWrapper(SpeakingPolicy):
    """Wraps M8ActorCritic to implement SpeakingPolicy interface."""

    def __init__(self, model: M8ActorCritic, device: torch.device):
        self._model = model
        self._device = device

    def decide(self, features: Dict[str, Any]) -> str:
        # Conversation features → obs format → encode
        obs = _to_obs(features)
        x = encode_obs(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits, _ = self._model(x)
            idx = logits.argmax(dim=-1).item()
        return ACTION_NAMES[idx]


def _to_obs(features: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ConversationFeatures to MultiStepTurnEnv obs format."""
    state_map = {"idle": "idle", "listening": "listen", "speaking": "speak",
                 "repair": "repair", "clarification": "clarify"}
    state = features.get("conversation_state", "idle")
    silence_s = features.get("seconds_since_last_user_speech", 999.0)
    return {
        "state": state_map.get(state, "listen"),
        "attribution": "silence",
        "silence_duration_ms": silence_s * 1000.0,
        "has_queued_utterance": features.get("has_queued_output", False),
        "user_speaking": features.get("is_user_speaking", False),
        "agent_speaking": features.get("is_agent_speaking", False),
        "queue_depth": 1 if features.get("is_agent_deferred", False) else 0,
    }


def load_distilled(checkpoint_path: str) -> DistilledPolicyWrapper:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = M8DistilledPolicy(
        input_dim=config.get("input_dim", 21),
        hidden_dim=config.get("hidden_dim", 64),
        dropout=0.0,
        num_actions=config.get("num_actions", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return DistilledPolicyWrapper(model, device)


def load_ppo(checkpoint_path: str) -> PPOPolicyWrapper:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = M8ActorCritic(
        input_dim=config.get("input_dim", 13),
        hidden_dim=config.get("hidden_dim", 64),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["policy_state"])
    model.eval()
    return PPOPolicyWrapper(model, device)


def compare_policies(episodes, policies: Dict[str, SpeakingPolicy], reward_config):
    """Run all policies on all episodes and compare."""
    results = []
    for ep in episodes:
        ep_results = {}
        for name, policy in policies.items():
            result = run_episode(ep, policy, reward_config)
            ep_results[name] = result

        # Compute action match rates vs heuristic
        h_steps = ep_results["heuristic"]["steps"]
        comparisons = {}
        for name in policies:
            if name == "heuristic":
                continue
            other_steps = ep_results[name]["steps"]
            min_len = min(len(h_steps), len(other_steps))
            if min_len > 0:
                matches = sum(1 for i in range(min_len)
                              if h_steps[i]["action"] == other_steps[i]["action"])
                comparisons[name] = {
                    "action_match": round(matches / min_len, 4),
                    "heuristic_reward": ep_results["heuristic"]["total_reward"],
                    f"{name}_reward": ep_results[name]["total_reward"],
                }

        results.append({
            "episode_id": ep["id"],
            "policies": {name: {
                "reward": ep_results[name]["total_reward"],
                "actions": ep_results[name]["action_counts"],
            } for name in policies},
            "comparisons": comparisons,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="M8c evaluation")
    parser.add_argument("--ppo-checkpoint", type=str, default="results/m8_ppo/best_policy.pt")
    parser.add_argument("--distilled-checkpoint", type=str, default="data/m8_distilled_policy.pt")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/m8_eval.json")
    parser.add_argument("--gate", action="store_true", default=True,
                        help="Run M8c acceptance gate")
    args = parser.parse_args()

    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        episodes = create_default_episodes()

    reward_config = M8RewardConfig()

    # Load policies
    policies: Dict[str, SpeakingPolicy] = {
        "heuristic": HeuristicSpeakingPolicy(),
        "distilled": load_distilled(args.distilled_checkpoint),
    }

    if Path(args.ppo_checkpoint).exists():
        policies["ppo"] = load_ppo(args.ppo_checkpoint)
    else:
        print(f"WARNING: PPO checkpoint not found: {args.ppo_checkpoint}")
        print("Skipping PPO evaluation")

    print("=" * 60)
    print("M8c Evaluation")
    print("=" * 60)

    results = compare_policies(episodes, policies, reward_config)

    # Print results
    for r in results:
        print(f"\n  {r['episode_id']}")
        for name, data in r["policies"].items():
            print(f"    {name:12s}: reward={data['reward']:+.4f}  actions={data['actions']}")
        for name, comp in r["comparisons"].items():
            print(f"    {name} vs heuristic: match={comp['action_match']:.4f}  "
                  f"reward_diff={comp[f'{name}_reward'] - comp['heuristic_reward']:+.4f}")

    # M8c gate: PPO vs distilled comparison
    if "ppo" in policies:
        print("\n" + "=" * 60)
        print("M8c Acceptance Gate (PPO vs Distilled)")
        print("=" * 60)

        expanded_eps = [e for e in episodes if e["id"] in
                        {"yield_on_interrupt", "resume_after_yield", "repair_after_collision",
                         "user_silence_then_speak"}]

        # Get rewards from results
        expanded_results = [r for r in results if r["episode_id"] in
                           {e["id"] for e in expanded_eps}]

        ppo_rewards = [r["policies"]["ppo"]["reward"] for r in expanded_results]
        dist_rewards = [r["policies"]["distilled"]["reward"] for r in expanded_results]

        if ppo_rewards and dist_rewards:
            avg_ppo = sum(ppo_rewards) / len(ppo_rewards)
            avg_dist = sum(dist_rewards) / len(dist_rewards)

            # Gate: PPO must beat distilled by >= 10%
            # Use absolute improvement for positive rewards, relative for negative
            if abs(avg_dist) > 0.01:
                improvement = (avg_ppo - avg_dist) / abs(avg_dist)
            else:
                improvement = avg_ppo - avg_dist

            passed = improvement >= MIN_PPO_IMPROVEMENT

            print(f"\n  Distilled avg reward (expanded): {avg_dist:+.4f}")
            print(f"  PPO avg reward (expanded):       {avg_ppo:+.4f}")
            print(f"  Improvement:                     {improvement:+.4f} ({improvement*100:+.1f}%)")
            print(f"  Target:                          >= {MIN_PPO_IMPROVEMENT:.0%}")
            print(f"\n  M8c ACCEPTANCE: {'PASS' if passed else 'FAIL'}")

            # Also show canonical
            canonical_results = [r for r in results if r["episode_id"] in CANONICAL_EPISODES]
            if canonical_results:
                ppo_canonical = sum(r["policies"]["ppo"]["reward"] for r in canonical_results) / len(canonical_results)
                dist_canonical = sum(r["policies"]["distilled"]["reward"] for r in canonical_results) / len(canonical_results)
                print(f"\n  Canonical — Distilled: {dist_canonical:+.4f}  PPO: {ppo_canonical:+.4f}")
        else:
            print("\n  WARNING: No PPO data for gate evaluation")
            passed = False
    else:
        passed = False

    # Save
    with open(args.output, "w") as f:
        json.dump({"results": results, "m8c_pass": passed}, f, indent=2)
    print(f"\nSaved to {args.output}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
