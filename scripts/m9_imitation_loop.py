"""M9a — Sound imitation trial runner.

Runs a simple motor-learning loop:
1. Generate a target sound (pure tone)
2. Producer generates a candidate sound
3. Compute acoustic similarity
4. Update controller weights (REINFORCE-like)
5. Repeat and track convergence

Usage:
    .venv/bin/python scripts/m9_imitation_loop.py --trials 20 --target-freq 440
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound.oscillator import OscillatorBank
from m9_sound.substrate import AcousticSubstrate
from m9_sound.controller import NeuralController, ControllerParams
from m9_sound.similarity import AcousticSimilarity
from m9_sound.producer import AcousticProducer, ProductionResult


def make_target(freq_hz: float, duration_s: float = 1.0,
                sr: int = 16000, waveform: str = "sine") -> np.ndarray:
    """Generate a target sound."""
    osc = OscillatorBank(sr=sr)
    if waveform == "triangle":
        return osc.triangle(freq_hz, duration_s, amplitude=0.5)
    elif waveform == "sawtooth":
        return osc.sawtooth(freq_hz, duration_s, amplitude=0.5)
    elif waveform == "square":
        return osc.square(freq_hz, duration_s, amplitude=0.5)
    else:
        return osc.sine(freq_hz, duration_s, amplitude=0.5)


def reinforce_step(controller: NeuralController, hidden: np.ndarray,
                   reward: float, baseline: float,
                   lr: float = 0.1) -> None:
    """Simple REINFORCE update: increase weights for actions above baseline.

    Parameters
    ----------
    controller : NeuralController
        The controller to update.
    hidden : np.ndarray
        Hidden state that was used for the decision.
    reward : float
        The reward received (similarity score).
    baseline : float
        Running average reward.
    lr : float
        Learning rate.
    """
    advantage = reward - baseline
    if advantage <= 0:
        return  # No update for below-average performance

    # Gradient of log-prob w.r.t. weights (simplified — linear readout)
    # For linear output: d(output)/d(W) = h, d(output)/d(b) = 1
    h = hidden.flatten()[:controller._hidden_dim]
    if len(h) < controller._hidden_dim:
        h = np.pad(h, (0, controller._hidden_dim - len(h)))

    # Update weights to increase probability of this action
    controller._W += lr * advantage * np.outer(np.ones(4), h)
    controller._b += lr * advantage * np.ones(4) * 0.5
    controller.clip_weights(max_norm=5.0)


def run_trial(producer: AcousticProducer, target: np.ndarray,
              trial_idx: int) -> Dict[str, Any]:
    """Run one imitation trial."""
    result = producer.produce_and_score(target)
    return {
        "trial": trial_idx,
        "similarity": result.similarity.overall,
        "mfcc_dist": result.similarity.mfcc_distance,
        "pitch_dist": result.similarity.pitch_distance,
        "duration_dist": result.similarity.duration_distance,
        "loudness_dist": result.similarity.loudness_distance,
        "freq_hz": result.params.freq_hz,
        "amplitude": result.params.amplitude,
        "waveform": result.params.waveform,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9a sound imitation loop")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--target-freq", type=float, default=440.0)
    parser.add_argument("--target-waveform", type=str, default="sine",
                        choices=["sine", "triangle", "sawtooth", "square"])
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--output", type=str, default="/tmp/m9_imitation_results.json")
    parser.add_argument("--waveforms", action="store_true",
                        help="Test all 4 waveform types")
    args = parser.parse_args()

    print("=" * 60)
    print("M9a — Sound Imitation Loop")
    print("=" * 60)
    print(f"  Target: {args.target_freq} Hz {args.target_waveform}, {args.duration}s")
    print(f"  Trials: {args.trials}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Learning rate: {args.lr}")
    print()

    producer = AcousticProducer(sr=args.sr, hidden_dim=args.hidden_dim)
    target = make_target(args.target_freq, args.duration, args.sr, args.target_waveform)

    results = []
    baseline = 0.0

    for trial in range(args.trials):
        result = run_trial(producer, target, trial)

        # Update baseline (running average)
        baseline = 0.9 * baseline + 0.1 * result["similarity"]

        # REINFORCE update
        reinforce_step(
            producer.controller,
            producer.substrate.hidden_state,
            result["similarity"],
            baseline,
            lr=args.lr,
        )

        results.append(result)

        if (trial + 1) % 5 == 0:
            recent = [r["similarity"] for r in results[-5:]]
            print(f"  Trial {trial+1:3d}/{args.trials}  "
                  f"sim={result['similarity']:.4f}  "
                  f"freq={result['freq_hz']:.1f}Hz  "
                  f"recent_avg={np.mean(recent):.4f}  "
                  f"baseline={baseline:.4f}")

    # Summary
    sims = [r["similarity"] for r in results]
    improvement = sims[-1] - sims[0] if len(sims) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Initial similarity: {sims[0]:.4f}")
    print(f"  Final similarity:   {sims[-1]:.4f}")
    print(f"  Improvement:        {improvement:+.4f} ({improvement/max(sims[0],0.001)*100:+.1f}%)")
    print(f"  Best similarity:    {max(sims):.4f} (trial {np.argmax(sims)})")
    print(f"  Mean similarity:    {np.mean(sims):.4f}")
    print(f"  Std deviation:      {np.std(sims):.4f}")

    # Save
    output = {
        "config": {
            "target_freq": args.target_freq,
            "target_waveform": args.target_waveform,
            "duration": args.duration,
            "trials": args.trials,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
        },
        "results": results,
        "summary": {
            "initial_similarity": float(sims[0]),
            "final_similarity": float(sims[-1]),
            "improvement": float(improvement),
            "best_similarity": float(max(sims)),
            "mean_similarity": float(np.mean(sims)),
            "std_similarity": float(np.std(sims)),
        }
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")

    # Acceptance criteria
    if improvement > 0.05:
        print(f"\n  ACCEPTANCE: PASS (improvement > 0.05)")
    else:
        print(f"\n  ACCEPTANCE: FAIL (improvement <= 0.05)")


if __name__ == "__main__":
    main()
