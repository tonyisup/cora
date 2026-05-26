#!/usr/bin/env python3
"""M8b — Benchmark + acceptance gate for distilled speaking policy.

Compares the distilled MLP policy against the heuristic baseline
on the scripted episode suite and checks M8b acceptance criteria.

Usage:
    # Benchmark only
    .venv/bin/python scripts/m8_benchmark.py --check

    # Acceptance gate (default)
    .venv/bin/python scripts/m8_benchmark.py --checkpoint data/m8_distilled_policy.pt

    # Both
    .venv/bin/python scripts/m8_benchmark.py --checkpoint data/m8_distilled_policy.pt --check --gate
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

from pfc.conversation_tracker import ConversationTracker
from pfc.speaking_policy import HeuristicSpeakingPolicy, SpeakingPolicy, ACTION_NAMES
from m8_policy.model import M8DistilledPolicy, encode_m8_features
from m8_conversation_simulator import create_default_episodes, load_episodes, run_episode
from m8_reward import M8RewardConfig


CANONICAL_EPISODES = {"speak_on_silence", "wait_while_user_speaks"}
EXPANDED_EPISODES = {"yield_on_interrupt", "resume_after_yield", "repair_after_collision", "user_silence_then_speak"}

MIN_CANONICAL_MATCH = 0.90
MIN_EXPANDED_MATCH = 0.80


class DistilledPolicyWrapper(SpeakingPolicy):
    """Wraps M8DistilledPolicy to implement the SpeakingPolicy interface."""

    def __init__(self, model: M8DistilledPolicy, device: torch.device):
        self._model = model
        self._device = device

    def decide(self, features: Dict[str, Any]) -> str:
        x = encode_m8_features(features).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            idx = logits.argmax(dim=-1).item()
        return ACTION_NAMES[idx]


def load_distilled_policy(checkpoint_path: str) -> DistilledPolicyWrapper:
    """Load a trained distilled policy from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = M8DistilledPolicy(
        input_dim=config.get("input_dim", 21),
        hidden_dim=config.get("hidden_dim", 64),
        dropout=config.get("dropout", 0.1),
        num_actions=config.get("num_actions", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return DistilledPolicyWrapper(model, device)


def compute_action_match(episodes, heuristic, distilled, reward_config):
    """Compute per-episode action match rate between two policies."""
    results = []
    for ep in episodes:
        h_result = run_episode(ep, heuristic, reward_config)
        d_result = run_episode(ep, distilled, reward_config)

        h_actions = h_result["steps"]
        d_actions = d_result["steps"]

        min_len = min(len(h_actions), len(d_actions))
        if min_len == 0:
            continue

        matches = sum(1 for i in range(min_len) if h_actions[i]["action"] == d_actions[i]["action"])
        match_rate = matches / min_len

        results.append({
            "episode_id": ep["id"],
            "match_rate": round(match_rate, 4),
            "steps": min_len,
            "heuristic_reward": h_result["total_reward"],
            "distilled_reward": d_result["total_reward"],
        })

    return results


def run_benchmark(checkpoint_path, episodes, reward_config):
    """Run full benchmark comparing heuristic vs distilled."""
    heuristic = HeuristicSpeakingPolicy()
    distilled = load_distilled_policy(checkpoint_path)

    print("=" * 60)
    print("M8b Benchmark — Distilled vs Heuristic")
    print("=" * 60)

    results = []
    for ep in episodes:
        h_result = run_episode(ep, heuristic, reward_config)
        d_result = run_episode(ep, distilled, reward_config)

        match = h_result["total_reward"] > 0 or d_result["total_reward"] >= h_result["total_reward"] * 0.9
        results.append({
            "episode_id": ep["id"],
            "heuristic_reward": h_result["total_reward"],
            "distilled_reward": d_result["total_reward"],
            "ratio": round(d_result["total_reward"] / max(abs(h_result["total_reward"]), 0.001), 4),
            "heuristic_actions": h_result["action_counts"],
            "distilled_actions": d_result["action_counts"],
            "match": match,
        })

        print(f"\n  {ep['id']}")
        print(f"    Heuristic:  reward={h_result['total_reward']:+.4f}  actions={h_result['action_counts']}")
        print(f"    Distilled:  reward={d_result['total_reward']:+.4f}  actions={d_result['action_counts']}")

    total_h = sum(r["heuristic_reward"] for r in results)
    total_d = sum(r["distilled_reward"] for r in results)
    all_match = all(r["match"] for r in results)

    print("\n" + "=" * 60)
    print(f"Total heuristic reward: {total_h:.4f}")
    print(f"Total distilled reward: {total_d:.4f}")
    print(f"Ratio: {total_d / max(abs(total_h), 0.001):.4f}")
    print(f"All episodes match: {all_match}")
    print("=" * 60)

    return {"episodes": results, "total_heuristic": total_h,
            "total_distilled": total_d, "all_match": all_match}


def run_acceptance_gate(checkpoint_path, episodes, reward_config):
    """Run M8b acceptance gate."""
    heuristic = HeuristicSpeakingPolicy()
    distilled = load_distilled_policy(checkpoint_path)

    ep_map = {ep["id"]: ep for ep in episodes}

    print("=" * 60)
    print("M8b Acceptance Gate")
    print("=" * 60)

    # Canonical
    canonical_eps = [ep_map[eid] for eid in CANONICAL_EPISODES if eid in ep_map]
    canonical_results = compute_action_match(canonical_eps, heuristic, distilled, reward_config)
    canonical_match = sum(r["match_rate"] for r in canonical_results) / max(len(canonical_results), 1)

    print(f"\nCanonical episodes (target >= {MIN_CANONICAL_MATCH:.0%}):")
    for r in canonical_results:
        status = "PASS" if r["match_rate"] >= MIN_CANONICAL_MATCH else "FAIL"
        print(f"  [{status}] {r['episode_id']:30s}  match={r['match_rate']:.4f}")
    print(f"  Average: {canonical_match:.4f}")

    # Expanded
    expanded_eps = [ep_map[eid] for eid in EXPANDED_EPISODES if eid in ep_map]
    expanded_results = compute_action_match(expanded_eps, heuristic, distilled, reward_config)
    expanded_match = sum(r["match_rate"] for r in expanded_results) / max(len(expanded_results), 1)

    print(f"\nExpanded episodes (target >= {MIN_EXPANDED_MATCH:.0%}):")
    for r in expanded_results:
        status = "PASS" if r["match_rate"] >= MIN_EXPANDED_MATCH else "FAIL"
        print(f"  [{status}] {r['episode_id']:30s}  match={r['match_rate']:.4f}")
    print(f"  Average: {expanded_match:.4f}")

    canonical_pass = canonical_match >= MIN_CANONICAL_MATCH
    expanded_pass = expanded_match >= MIN_EXPANDED_MATCH
    all_pass = canonical_pass and expanded_pass

    print("\n" + "=" * 60)
    print(f"Canonical match: {canonical_match:.4f}  {'PASS' if canonical_pass else 'FAIL'}")
    print(f"Expanded match:  {expanded_match:.4f}  {'PASS' if expanded_pass else 'FAIL'}")
    print(f"\nM8b ACCEPTANCE: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="M8b benchmark + acceptance gate")
    parser.add_argument("--checkpoint", type=str, default="data/m8_distilled_policy.pt")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--output", type=str, default="/tmp/m8_benchmark.json")
    parser.add_argument("--check", action="store_true",
                        help="Run benchmark comparison")
    parser.add_argument("--gate", action="store_true",
                        help="Run acceptance gate")
    args = parser.parse_args()

    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        episodes = create_default_episodes()

    reward_config = M8RewardConfig()

    # Default: run gate if neither specified
    if not args.check and not args.gate:
        args.gate = True

    if args.check:
        result = run_benchmark(args.checkpoint, episodes, reward_config)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nBenchmark saved to {args.output}")

    if args.gate:
        passed = run_acceptance_gate(args.checkpoint, episodes, reward_config)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
