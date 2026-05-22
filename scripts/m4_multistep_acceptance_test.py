#!/usr/bin/env python3
"""M4 multi-step acceptance test.

Checks that the distilled baseline and the PPO checkpoint both preserve the
original one-step canonical/expanded suites and also behave sanely on the new
multi-step sequence benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is required. Run: .venv/bin/pip install torch")

from m4_eval import evaluate_policy as evaluate_one_step, load_policy
from m4_multistep_benchmark import evaluate as evaluate_multi_step


EXPECTED_ONE_STEP = 1.0
MAX_MULTI_STEP_DROP = 1e-4


def load_and_eval_one_step(policy_kind: str, checkpoint: str, device: torch.device) -> Dict[str, Any]:
    policy = load_policy(policy_kind, checkpoint, device)
    return evaluate_one_step(policy, device)


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 multi-step acceptance test")
    parser.add_argument("--distilled-checkpoint", required=True)
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distilled_one = load_and_eval_one_step("distilled", args.distilled_checkpoint, device)
    ppo_one = load_and_eval_one_step("ppo", args.ppo_checkpoint, device)
    distilled_multi = evaluate_multi_step(load_policy("distilled", args.distilled_checkpoint, device), 4, device)
    ppo_multi = evaluate_multi_step(load_policy("ppo", args.ppo_checkpoint, device), 4, device)

    failures = []
    if distilled_one["accuracy"] < EXPECTED_ONE_STEP:
        failures.append(f"Distilled one-step accuracy {distilled_one['accuracy']:.3f} < 1.000")
    if ppo_one["accuracy"] < EXPECTED_ONE_STEP:
        failures.append(f"PPO one-step accuracy {ppo_one['accuracy']:.3f} < 1.000")
    if distilled_one["category_accuracy"].get("canonical", 0.0) < EXPECTED_ONE_STEP:
        failures.append("Distilled one-step canonical accuracy below 100%")
    if ppo_one["category_accuracy"].get("canonical", 0.0) < EXPECTED_ONE_STEP:
        failures.append("PPO one-step canonical accuracy below 100%")

    if ppo_multi["mean_episode_return"] < distilled_multi["mean_episode_return"] - MAX_MULTI_STEP_DROP:
        failures.append(
            f"PPO multi-step return {ppo_multi['mean_episode_return']:.6f} < distilled {distilled_multi['mean_episode_return']:.6f}"
        )

    report = {
        "device": str(device),
        "distilled_one_step": distilled_one,
        "ppo_one_step": ppo_one,
        "distilled_multi_step": distilled_multi,
        "ppo_multi_step": ppo_multi,
        "failures": failures,
    }

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out}")

    if failures:
        for fail in failures:
            print(f"FAIL: {fail}", file=sys.stderr)
        return 1

    print("PASS: M4 multi-step acceptance criteria satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
