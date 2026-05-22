#!/usr/bin/env python3
"""M4 acceptance test.

Checks that the distilled baseline and the PPO checkpoint both satisfy the
canonical and expanded scenario suites, and that the PPO policy does not
regress on the canonical benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from m4_eval import evaluate_policy, load_policy

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is required. Run: .venv/bin/pip install torch")


EXPECTED_CANONICAL = 1.0
EXPECTED_EXPANDED = 1.0


def load_and_eval(policy_kind: str, checkpoint: str, device: torch.device) -> Dict[str, Any]:
    policy = load_policy(policy_kind, checkpoint, device)
    return evaluate_policy(policy, device)


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 acceptance test")
    parser.add_argument("--distilled-checkpoint", required=True)
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distilled = load_and_eval("distilled", args.distilled_checkpoint, device)
    ppo = load_and_eval("ppo", args.ppo_checkpoint, device)

    failures = []
    if distilled["accuracy"] < EXPECTED_EXPANDED:
        failures.append(f"Distilled accuracy {distilled['accuracy']:.3f} < {EXPECTED_EXPANDED:.3f}")
    if ppo["accuracy"] < EXPECTED_EXPANDED:
        failures.append(f"PPO accuracy {ppo['accuracy']:.3f} < {EXPECTED_EXPANDED:.3f}")
    if distilled["category_accuracy"].get("canonical", 0.0) < EXPECTED_CANONICAL:
        failures.append("Distilled canonical accuracy below 100%")
    if ppo["category_accuracy"].get("canonical", 0.0) < EXPECTED_CANONICAL:
        failures.append("PPO canonical accuracy below 100%")
    if ppo["total_reward"] < distilled["total_reward"] - 1e-6:
        failures.append(
            f"PPO total reward {ppo['total_reward']:.3f} < distilled {distilled['total_reward']:.3f}"
        )

    report = {
        "device": str(device),
        "distilled": distilled,
        "ppo": ppo,
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

    print("PASS: M4 acceptance criteria satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
