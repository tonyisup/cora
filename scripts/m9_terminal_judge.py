#!/usr/bin/env python3
"""M9c — Weak terminal confirmation.

Compares two learning loops on the same target and seed:
- similarity-only reward (baseline)
- combined acoustic + terminal reward

The terminal judge is intentionally weak: it is a thresholded proxy that can be
used automatically in tests or swapped out for a human binary verdict later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import (
    AcousticProducer,
    OscillatorBank,
    WeakTerminalJudge,
    attach_label_if_passed,
    combine_reward,
    summarize_audio,
    refine_prototype,
)
from sag import AcousticNode


def make_target(freq_hz: float, duration_s: float = 1.0, sr: int = 16000, waveform: str = "sine") -> np.ndarray:
    osc = OscillatorBank(sr=sr)
    fn = {
        "sine": osc.sine,
        "triangle": osc.triangle,
        "sawtooth": osc.sawtooth,
        "square": osc.square,
    }[waveform]
    return fn(freq_hz, duration_s, amplitude=0.5)


def reinforce_step(controller, hidden: np.ndarray, reward: float, baseline: float, lr: float) -> None:
    """A tiny REINFORCE-style update used by M9a/M9c loops."""
    advantage = reward - baseline
    if advantage <= 0:
        return

    h = hidden.flatten()[:controller._hidden_dim]
    if len(h) < controller._hidden_dim:
        h = np.pad(h, (0, controller._hidden_dim - len(h)))

    controller._W += lr * advantage * np.outer(np.ones(4), h)
    controller._b += lr * advantage * np.ones(4) * 0.5
    controller.clip_weights(max_norm=5.0)


def run_loop(
    *,
    producer: AcousticProducer,
    target: np.ndarray,
    target_label: str,
    trials: int,
    lr: float,
    judge: WeakTerminalJudge,
    use_terminal_reward: bool,
) -> Dict[str, Any]:
    target_summary = summarize_audio(target, producer._sr)
    prototype = AcousticNode.from_audio_summary(
        id=f"proto_{target_label}",
        core=f"Prototype for {target_label}",
        grounding=[target_label],
        label=None,
        context={"target_label": target_label},
        mfcc_centroid=target_summary["mfcc_centroid"],
        pitch_mean_hz=target_summary["pitch_hz"],
        duration_mean_s=target_summary["duration_s"],
        loudness_mean_db=target_summary["loudness_db"],
        source_trials=["target"],
    )

    baseline = 0.0
    rows: List[Dict[str, Any]] = []
    sims: List[float] = []
    terminal_passes = 0
    first_pass_trial: int | None = None

    for trial in range(trials):
        result = producer.produce_and_score(target)
        produced_summary = summarize_audio(result.audio, producer._sr)
        terminal = judge.judge(result.similarity, label=target_label)
        combined = combine_reward(result.similarity, terminal)

        reward = combined.value if use_terminal_reward else float(result.similarity.overall)
        baseline = 0.9 * baseline + 0.1 * reward
        reinforce_step(producer.controller, producer.substrate.hidden_state, reward, baseline, lr)

        if terminal.passed:
            terminal_passes += 1
            if first_pass_trial is None:
                first_pass_trial = trial + 1
            prototype = attach_label_if_passed(prototype, terminal)
            prototype = refine_prototype(
                prototype,
                trial_id=f"trial_{trial+1:03d}",
                similarity=float(result.similarity.overall),
                audio_summary=produced_summary,
            )
        else:
            prototype = refine_prototype(
                prototype,
                trial_id=f"trial_{trial+1:03d}",
                similarity=float(result.similarity.overall),
                audio_summary=produced_summary,
            )

        sims.append(float(result.similarity.overall))
        rows.append(
            {
                "trial": trial + 1,
                "similarity": float(result.similarity.overall),
                "reward": float(reward),
                "combined_reward": float(combined.value),
                "terminal_passed": terminal.passed,
                "terminal_reason": terminal.reason,
                "freq_hz": float(result.params.freq_hz),
                "waveform": int(result.params.waveform),
            }
        )

    return {
        "trials": rows,
        "summary": {
            "initial_similarity": float(sims[0]),
            "final_similarity": float(sims[-1]),
            "improvement": float(sims[-1] - sims[0]),
            "mean_similarity": float(np.mean(sims)),
            "std_similarity": float(np.std(sims)),
            "terminal_pass_rate": float(terminal_passes / max(trials, 1)),
            "first_terminal_pass_trial": first_pass_trial,
            "prototype_label": prototype.label,
            "prototype_grounding": prototype.grounding,
            "prototype_avg_similarity": float(prototype.avg_similarity),
            "prototype_production_count": int(prototype.production_count),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9c weak terminal confirmation")
    parser.add_argument("--target-freq", type=float, default=440.0)
    parser.add_argument("--target-waveform", type=str, default="sine",
                        choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-threshold", type=float, default=0.54)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--terminal-weight", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="/tmp/m9_terminal_confirmation.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    target = make_target(args.target_freq, args.duration, args.sr, args.target_waveform)
    target_label = f"{int(round(args.target_freq))}_{args.target_waveform}"
    judge = WeakTerminalJudge(threshold=args.judge_threshold)

    print("=" * 72)
    print("M9c — Weak Terminal Confirmation")
    print("=" * 72)
    print(f"  Target: {args.target_freq:.1f} Hz {args.target_waveform}")
    print(f"  Trials: {args.trials}")
    print(f"  Judge threshold: {args.judge_threshold:.3f}")
    print(f"  Reward weights: similarity={args.similarity_weight:.2f}, terminal={args.terminal_weight:.2f}")
    print(f"  Seed: {args.seed}")
    print()

    np.random.seed(args.seed)
    baseline_producer = AcousticProducer(sr=args.sr, hidden_dim=args.hidden_dim)
    baseline = run_loop(
        producer=baseline_producer,
        target=target,
        target_label=target_label,
        trials=args.trials,
        lr=args.lr,
        judge=judge,
        use_terminal_reward=False,
    )

    np.random.seed(args.seed)
    terminal_producer = AcousticProducer(sr=args.sr, hidden_dim=args.hidden_dim)
    terminal = run_loop(
        producer=terminal_producer,
        target=target,
        target_label=target_label,
        trials=args.trials,
        lr=args.lr,
        judge=judge,
        use_terminal_reward=True,
    )

    baseline_improvement = baseline["summary"]["improvement"]
    terminal_improvement = terminal["summary"]["improvement"]
    speedup = terminal["summary"]["first_terminal_pass_trial"] or args.trials + 1

    verdict = terminal["summary"]["terminal_pass_rate"] > 0.70

    output = {
        "config": {
            "target_freq": args.target_freq,
            "target_waveform": args.target_waveform,
            "duration": args.duration,
            "trials": args.trials,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "judge_threshold": args.judge_threshold,
            "similarity_weight": args.similarity_weight,
            "terminal_weight": args.terminal_weight,
        },
        "baseline": baseline,
        "terminal": terminal,
        "comparison": {
            "baseline_improvement": baseline_improvement,
            "terminal_improvement": terminal_improvement,
            "terminal_first_pass_trial": speedup,
            "improvement_delta": float(terminal_improvement - baseline_improvement),
            "pass_rate_delta": float(terminal["summary"]["terminal_pass_rate"] - baseline["summary"]["terminal_pass_rate"]),
            "verdict": verdict,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))

    print("Baseline:")
    print(
        f"  final_sim={baseline['summary']['final_similarity']:.4f} "
        f"improvement={baseline['summary']['improvement']:+.4f} "
        f"mean={baseline['summary']['mean_similarity']:.4f} "
        f"pass_rate={baseline['summary']['terminal_pass_rate']:.2f}"
    )
    print("Terminal-confirmed:")
    print(
        f"  final_sim={terminal['summary']['final_similarity']:.4f} "
        f"improvement={terminal['summary']['improvement']:+.4f} "
        f"mean={terminal['summary']['mean_similarity']:.4f} "
        f"pass_rate={terminal['summary']['terminal_pass_rate']:.2f}"
    )
    print("Comparison:")
    print(f"  improvement_delta={output['comparison']['improvement_delta']:+.4f}")
    print(f"  pass_rate_delta={output['comparison']['pass_rate_delta']:+.2f}")
    print(f"  first_terminal_pass_trial={speedup}")
    print(f"  Verdict: {'PASS' if verdict else 'FAIL'}")
    print(f"  Saved to: {args.output}")

    if not verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
