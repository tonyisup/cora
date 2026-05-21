#!/usr/bin/env python3
"""M2 — Evaluation for Distilled Speaking Policy.

Evaluates a trained checkpoint on:
1. The collected JSONL datasets (deterministic / teacher / hard negatives)
2. The six scenario-level benchmark cases from M1

This is a lightweight evaluation harness for comparing the distilled model
against the deterministic baseline on the same action space.

Usage:
  .venv/bin/python scripts/m2_eval.py --checkpoint results/m2_model_checkpoints/distilled_v2/best_model.pt
  .venv/bin/python scripts/m2_eval.py --checkpoint results/m2_model_checkpoints/distilled_v2/best_model.pt --output results/m2_eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch is required. Run: pip install torch")

from m2_train import (
    ACTION_INDEX,
    ATTR_INDEX,
    NUM_ACTIONS,
    STATE_INDEX,
    DistilledSpeakingPolicy,
    SpeakingPolicyDataset,
    encode_features,
)
from m2_guarded_policy import GuardedDistilledPolicy


SCENARIO_SUITE: List[Dict[str, Any]] = [
    {
        "name": "speak_on_silence",
        "expected": "speak",
        "features": {
            "state": "listen",
            "state_idx": STATE_INDEX["listen"],
            "attribution": "silence",
            "attr_idx": ATTR_INDEX["silence"],
            "silence_duration_ms": 400.0,
            "silence_duration_log": np.log1p(400.0 / 1000.0),
            "vad_energy_db": -55.0,
            "has_queued_utterance": True,
            "user_speaking": False,
            "agent_speaking": False,
            "asr_text": "",
            "speaker": "",
        },
    },
    {
        "name": "wait_while_user_speaks",
        "expected": "wait",
        "features": {
            "state": "listen",
            "state_idx": STATE_INDEX["listen"],
            "attribution": "world",
            "attr_idx": ATTR_INDEX["world"],
            "silence_duration_ms": 120.0,
            "silence_duration_log": np.log1p(120.0 / 1000.0),
            "vad_energy_db": -20.0,
            "has_queued_utterance": True,
            "user_speaking": True,
            "agent_speaking": False,
            "asr_text": "I'm still talking about the account issue.",
            "speaker": "user",
        },
    },
    {
        "name": "yield_on_interrupt",
        "expected": "yield",
        "features": {
            "state": "speak",
            "state_idx": STATE_INDEX["speak"],
            "attribution": "world",
            "attr_idx": ATTR_INDEX["world"],
            "silence_duration_ms": 0.0,
            "silence_duration_log": np.log1p(0.0),
            "vad_energy_db": -18.0,
            "has_queued_utterance": False,
            "user_speaking": True,
            "agent_speaking": True,
            "asr_text": "Sorry to interrupt, but I need help right now.",
            "speaker": "user",
        },
    },
    {
        "name": "yield_on_collision",
        "expected": "yield",
        "features": {
            "state": "speak",
            "state_idx": STATE_INDEX["speak"],
            "attribution": "both",
            "attr_idx": ATTR_INDEX["both"],
            "silence_duration_ms": 0.0,
            "silence_duration_log": np.log1p(0.0),
            "vad_energy_db": -15.0,
            "has_queued_utterance": False,
            "user_speaking": True,
            "agent_speaking": True,
            "asr_text": "(overlap)",
            "speaker": "",
        },
    },
    {
        "name": "clarify_after_silence",
        "expected": "clarify",
        "features": {
            "state": "listen",
            "state_idx": STATE_INDEX["listen"],
            "attribution": "silence",
            "attr_idx": ATTR_INDEX["silence"],
            "silence_duration_ms": 2500.0,
            "silence_duration_log": np.log1p(2500.0 / 1000.0),
            "vad_energy_db": -55.0,
            "has_queued_utterance": False,
            "user_speaking": False,
            "agent_speaking": False,
            "asr_text": "Yes, it keeps making a weird noise.",
            "speaker": "",
        },
    },
    {
        "name": "repair_in_repair_state",
        "expected": "repair",
        "features": {
            "state": "repair",
            "state_idx": STATE_INDEX["repair"],
            "attribution": "silence",
            "attr_idx": ATTR_INDEX["silence"],
            "silence_duration_ms": 0.0,
            "silence_duration_log": np.log1p(0.0),
            "vad_energy_db": -60.0,
            "has_queued_utterance": False,
            "user_speaking": False,
            "agent_speaking": False,
            "asr_text": "",
            "speaker": "",
        },
    },
]


class EvalPolicy:
    def __init__(self, model: DistilledSpeakingPolicy):
        self.model = model

    def predict(self, features: Dict[str, Any]) -> str:
        return self.model.predict(features)[0]


def evaluate_dataset(model: DistilledSpeakingPolicy, paths: List[str], device: torch.device) -> Dict[str, Any]:
    dataset = SpeakingPolicyDataset(paths)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    model.eval()

    total = 0
    correct = 0
    per_action_total = {a: 0 for a in range(NUM_ACTIONS)}
    per_action_correct = {a: 0 for a in range(NUM_ACTIONS)}

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            actions = batch["action"].to(device)
            logits = model(features)
            preds = logits.argmax(dim=-1)
            total += features.size(0)
            correct += (preds == actions).sum().item()
            for a in range(NUM_ACTIONS):
                mask = actions == a
                per_action_total[a] += mask.sum().item()
                per_action_correct[a] += ((preds == a) & mask).sum().item()

    action_names = ["speak", "wait", "clarify", "repair", "yield"]
    per_action_acc = {
        action_names[a]: (per_action_correct[a] / per_action_total[a]) if per_action_total[a] else None
        for a in range(NUM_ACTIONS)
    }

    return {
        "accuracy": correct / total if total else 0.0,
        "per_action_accuracy": per_action_acc,
        "samples": total,
    }


def evaluate_scenarios(policy: Any) -> Dict[str, Any]:
    results = []
    correct = 0
    for scenario in SCENARIO_SUITE:
        pred_out = policy.predict(scenario["features"])
        pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        ok = pred == scenario["expected"]
        correct += int(ok)
        results.append({
            "name": scenario["name"],
            "expected": scenario["expected"],
            "predicted": pred,
            "correct": ok,
        })
    return {
        "accuracy": correct / len(SCENARIO_SUITE),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate distilled speaking policy")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", nargs="*", default=["results/m2_data_deterministic.jsonl", "results/m2_data_teacher.jsonl", "results/m2_data_hardneg.jsonl"], help="JSONL datasets to evaluate")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--guarded", action="store_true", help="Evaluate scenario suite with deterministic safety rails")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = DistilledSpeakingPolicy()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    dataset_report = evaluate_dataset(model, args.data, device)
    policy = GuardedDistilledPolicy(model) if args.guarded else model
    scenario_report = evaluate_scenarios(policy)

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "guarded": args.guarded,
        "dataset_report": dataset_report,
        "scenario_report": scenario_report,
        "checkpoint_meta": {
            "epoch": ckpt.get("epoch"),
            "val_accuracy": ckpt.get("val_accuracy"),
            "args": ckpt.get("args", {}),
        },
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
