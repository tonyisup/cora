#!/usr/bin/env python3
"""M3 — Expanded scenario benchmark for the speaking policy.

This extends the original six canonical scenarios with additional ambiguous
cases that specifically probe clarify / wait / yield / repair boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is required. Run: pip install torch")

from m2_train import (
    ATTR_INDEX,
    NUM_ACTIONS,
    STATE_INDEX,
    DistilledSpeakingPolicy,
)
from m2_guarded_policy import GuardedDistilledPolicy


ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]


def feature_template(
    *,
    state: str,
    attribution: str,
    silence_duration_ms: float,
    vad_energy_db: float,
    has_queued_utterance: bool,
    user_speaking: bool,
    agent_speaking: bool,
    asr_text: str,
    speaker: str,
) -> Dict[str, Any]:
    return {
        "state": state,
        "state_idx": STATE_INDEX[state],
        "attribution": attribution,
        "attr_idx": ATTR_INDEX[attribution],
        "silence_duration_ms": silence_duration_ms,
        "silence_duration_log": np.log1p(silence_duration_ms / 1000.0),
        "vad_energy_db": vad_energy_db,
        "has_queued_utterance": has_queued_utterance,
        "user_speaking": user_speaking,
        "agent_speaking": agent_speaking,
        "asr_text": asr_text,
        "speaker": speaker,
    }


SCENARIO_SUITE: List[Dict[str, Any]] = [
    {
        "name": "speak_on_silence",
        "expected": "speak",
        "category": "canonical",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=400.0,
            vad_energy_db=-55.0,
            has_queued_utterance=True,
            user_speaking=False,
            agent_speaking=False,
            asr_text="",
            speaker="",
        ),
    },
    {
        "name": "wait_while_user_speaks",
        "expected": "wait",
        "category": "canonical",
        "features": feature_template(
            state="listen",
            attribution="world",
            silence_duration_ms=120.0,
            vad_energy_db=-20.0,
            has_queued_utterance=True,
            user_speaking=True,
            agent_speaking=False,
            asr_text="I'm still talking about the account issue.",
            speaker="user",
        ),
    },
    {
        "name": "yield_on_interrupt",
        "expected": "yield",
        "category": "canonical",
        "features": feature_template(
            state="speak",
            attribution="world",
            silence_duration_ms=0.0,
            vad_energy_db=-18.0,
            has_queued_utterance=False,
            user_speaking=True,
            agent_speaking=True,
            asr_text="Sorry to interrupt, but I need help right now.",
            speaker="user",
        ),
    },
    {
        "name": "yield_on_collision",
        "expected": "yield",
        "category": "canonical",
        "features": feature_template(
            state="speak",
            attribution="both",
            silence_duration_ms=0.0,
            vad_energy_db=-15.0,
            has_queued_utterance=False,
            user_speaking=True,
            agent_speaking=True,
            asr_text="(overlap)",
            speaker="",
        ),
    },
    {
        "name": "clarify_after_silence",
        "expected": "clarify",
        "category": "canonical",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=2500.0,
            vad_energy_db=-55.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="Yes, it keeps making a weird noise.",
            speaker="",
        ),
    },
    {
        "name": "repair_in_repair_state",
        "expected": "repair",
        "category": "canonical",
        "features": feature_template(
            state="repair",
            attribution="silence",
            silence_duration_ms=0.0,
            vad_energy_db=-60.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="",
            speaker="",
        ),
    },
    {
        "name": "wait_in_idle_no_queue",
        "expected": "wait",
        "category": "ambiguous",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=150.0,
            vad_energy_db=-58.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="",
            speaker="",
        ),
    },
    {
        "name": "speak_on_short_silence_with_queue",
        "expected": "speak",
        "category": "ambiguous",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=320.0,
            vad_energy_db=-54.0,
            has_queued_utterance=True,
            user_speaking=False,
            agent_speaking=False,
            asr_text="",
            speaker="",
        ),
    },
    {
        "name": "yield_on_long_interrupt_after_speech",
        "expected": "yield",
        "category": "ambiguous",
        "features": feature_template(
            state="speak",
            attribution="world",
            silence_duration_ms=0.0,
            vad_energy_db=-12.0,
            has_queued_utterance=True,
            user_speaking=True,
            agent_speaking=True,
            asr_text="Actually, sorry, let me stop you there — my issue is different.",
            speaker="user",
        ),
    },
    {
        "name": "yield_on_overlap_with_queue",
        "expected": "yield",
        "category": "ambiguous",
        "features": feature_template(
            state="speak",
            attribution="both",
            silence_duration_ms=20.0,
            vad_energy_db=-16.0,
            has_queued_utterance=True,
            user_speaking=True,
            agent_speaking=True,
            asr_text="(overlap, queued reply exists)",
            speaker="",
        ),
    },
    {
        "name": "clarify_after_long_silence_no_queue",
        "expected": "clarify",
        "category": "ambiguous",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=3200.0,
            vad_energy_db=-57.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="Yes, the thing I mentioned earlier still isn't fixed.",
            speaker="",
        ),
    },
    {
        "name": "clarify_after_question_marking",
        "expected": "clarify",
        "category": "ambiguous",
        "features": feature_template(
            state="listen",
            attribution="silence",
            silence_duration_ms=2800.0,
            vad_energy_db=-56.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="...maybe? I’m not sure.",
            speaker="",
        ),
    },
    {
        "name": "repair_after_failed_repair_prompt",
        "expected": "repair",
        "category": "ambiguous",
        "features": feature_template(
            state="repair",
            attribution="silence",
            silence_duration_ms=0.0,
            vad_energy_db=-60.0,
            has_queued_utterance=False,
            user_speaking=False,
            agent_speaking=False,
            asr_text="",
            speaker="",
        ),
    },
]


def evaluate_dataset(model: DistilledSpeakingPolicy, paths: List[str], device: Any) -> Dict[str, Any]:
    from m2_train import SpeakingPolicyDataset

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

    per_action_acc = {
        ACTION_NAMES[a]: (per_action_correct[a] / per_action_total[a]) if per_action_total[a] else None
        for a in range(NUM_ACTIONS)
    }

    return {
        "accuracy": correct / total if total else 0.0,
        "per_action_accuracy": per_action_acc,
        "samples": total,
    }


class EvalPolicy:
    def __init__(self, model: DistilledSpeakingPolicy):
        self.model = model

    def predict(self, features: Dict[str, Any]) -> str:
        return self.model.predict(features)[0]


def evaluate_scenarios(policy: Any) -> Dict[str, Any]:
    results = []
    correct = 0
    by_category = {}
    for scenario in SCENARIO_SUITE:
        pred_out = policy.predict(scenario["features"])
        pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        ok = pred == scenario["expected"]
        correct += int(ok)
        by_category.setdefault(scenario["category"], {"correct": 0, "total": 0})
        by_category[scenario["category"]]["total"] += 1
        by_category[scenario["category"]]["correct"] += int(ok)
        results.append(
            {
                "name": scenario["name"],
                "category": scenario["category"],
                "expected": scenario["expected"],
                "predicted": pred,
                "correct": ok,
            }
        )
    category_accuracy = {
        cat: (vals["correct"] / vals["total"] if vals["total"] else 0.0)
        for cat, vals in by_category.items()
    }
    return {
        "accuracy": correct / len(SCENARIO_SUITE),
        "category_accuracy": category_accuracy,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate distilled speaking policy on expanded M3 scenarios")
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
    policy = GuardedDistilledPolicy(model) if args.guarded else EvalPolicy(model)
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
