#!/usr/bin/env python3
"""M9b — Repeatability and convergence evaluation.

Runs the M9 sound producer against one or more target sounds, measures
trial-to-trial repeatability, and refines acoustic prototype nodes in the SAG.

This script is deliberately modest: it validates that the motor-learning loop
is stable and that acoustic prototypes can be incrementally updated. It does
not claim full text association or terminal confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound.oscillator import OscillatorBank
from m9_sound.producer import AcousticProducer
from m9_sound.convergence import (
    compute_repeatability_stats,
    refine_prototype,
    summarize_audio,
)
from sag import AcousticNode, SemanticAbstractionGraph, similar_to_edge


def make_target(freq_hz: float, duration_s: float, sr: int, waveform: str) -> np.ndarray:
    osc = OscillatorBank(sr=sr)
    fn = {
        "sine": osc.sine,
        "triangle": osc.triangle,
        "sawtooth": osc.sawtooth,
        "square": osc.square,
    }[waveform]
    return fn(freq_hz, duration_s, amplitude=0.5)


def parse_targets(text: str) -> list[tuple[float, str]]:
    """Parse a target spec like '440:sine,550:triangle'."""
    out: list[tuple[float, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            freq_s, wf = item.split(":", 1)
            out.append((float(freq_s), wf.strip()))
        else:
            out.append((float(item), "sine"))
    return out


def run_target(
    *,
    producer: AcousticProducer,
    graph: SemanticAbstractionGraph,
    freq_hz: float,
    waveform: str,
    duration_s: float,
    trials: int,
    sr: int,
) -> dict[str, Any]:
    target = make_target(freq_hz=freq_hz, duration_s=duration_s, sr=sr, waveform=waveform)
    target_summary = summarize_audio(target, sr)

    proto_id = f"proto_{int(round(freq_hz))}_{waveform}"
    prototype = AcousticNode.from_audio_summary(
        id=proto_id,
        core=f"{freq_hz:.1f} Hz {waveform} target prototype",
        grounding=[f"{int(round(freq_hz))}hz", waveform, proto_id],
        context={"freq_hz": freq_hz, "waveform": waveform, "kind": "prototype"},
        label=proto_id,
        pitch_mean_hz=target_summary["pitch_hz"],
        duration_mean_s=target_summary["duration_s"],
        loudness_mean_db=target_summary["loudness_db"],
        mfcc_centroid=target_summary["mfcc_centroid"],
        source_trials=["target"],
    )
    graph.add_node(prototype)

    trial_rows: list[dict[str, Any]] = []
    similarities: list[float] = []

    for trial_idx in range(trials):
        trial_id = f"{proto_id}_trial_{trial_idx + 1:03d}"
        produced = producer.produce(duration_s)
        produced_summary = summarize_audio(produced, sr)
        sim = producer.similarity.detailed_score(target, produced)
        similarities.append(float(sim.overall))

        exemplar_id = f"{trial_id}_exemplar"
        exemplar = AcousticNode.from_audio_summary(
            id=exemplar_id,
            core=f"Exemplar from {proto_id}",
            grounding=[proto_id, waveform, f"trial_{trial_idx + 1:03d}"],
            context={"kind": "exemplar", "freq_hz": freq_hz, "waveform": waveform},
            pitch_mean_hz=produced_summary["pitch_hz"],
            duration_mean_s=produced_summary["duration_s"],
            loudness_mean_db=produced_summary["loudness_db"],
            mfcc_centroid=produced_summary["mfcc_centroid"],
            source_trials=[trial_id],
        )
        graph.add_node(exemplar)
        graph.add_edge(similar_to_edge(exemplar_id, proto_id, confidence=float(sim.overall)))

        prototype = refine_prototype(
            prototype,
            trial_id=trial_id,
            similarity=float(sim.overall),
            audio_summary=produced_summary,
        )
        graph.add_node(prototype)

        trial_rows.append(
            {
                "trial": trial_idx + 1,
                "similarity": float(sim.overall),
                "mfcc_distance": float(sim.mfcc_distance),
                "pitch_distance": float(sim.pitch_distance),
                "duration_distance": float(sim.duration_distance),
                "loudness_distance": float(sim.loudness_distance),
                "prototype_avg_similarity": float(prototype.avg_similarity),
                "prototype_production_count": int(prototype.production_count),
            }
        )

    stats = compute_repeatability_stats(similarities)
    resolved = graph.resolve(proto_id)

    return {
        "target": {
            "freq_hz": freq_hz,
            "waveform": waveform,
            "duration_s": duration_s,
            "prototype_id": proto_id,
            "target_summary": target_summary,
        },
        "trials": trial_rows,
        "stats": {
            "n_trials": stats.n_trials,
            "mean": stats.mean,
            "std": stats.std,
            "cv": stats.cv,
            "first": stats.first,
            "last": stats.last,
            "improvement": stats.improvement,
            "best": stats.best,
            "worst": stats.worst,
            "converged": stats.converged,
        },
        "prototype": {
            "resolved_by_label": resolved.id if resolved is not None else None,
            "production_count": prototype.production_count,
            "avg_similarity": prototype.avg_similarity,
            "source_trials": prototype.source_trials,
            "pitch_mean_hz": prototype.pitch_mean_hz,
            "duration_mean_s": prototype.duration_mean_s,
            "loudness_mean_db": prototype.loudness_mean_db,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9b repeatability and convergence test")
    parser.add_argument("--targets", type=str, default="440:sine",
                        help="Comma-separated target specs like '440:sine,550:triangle'")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output", type=str, default="/tmp/m9_convergence_results.json")
    parser.add_argument("--cv-threshold", type=float, default=0.2)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    if not targets:
        raise SystemExit("No targets specified")

    print("=" * 72)
    print("M9b — Repeatability and Convergence")
    print("=" * 72)
    print(f"  Targets: {', '.join(f'{f:.1f}:{wf}' for f, wf in targets)}")
    print(f"  Trials per target: {args.trials}")
    print(f"  Duration: {args.duration:.2f}s")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  CV threshold: {args.cv_threshold:.2f}")
    print()

    producer = AcousticProducer(sr=args.sr, hidden_dim=args.hidden_dim)
    graph = SemanticAbstractionGraph()
    results = []

    for freq_hz, waveform in targets:
        print(f"-- Target {freq_hz:.1f} Hz {waveform} --")
        result = run_target(
            producer=producer,
            graph=graph,
            freq_hz=freq_hz,
            waveform=waveform,
            duration_s=args.duration,
            trials=args.trials,
            sr=args.sr,
        )
        stats = result["stats"]
        print(
            f"  mean={stats['mean']:.4f} std={stats['std']:.4f} cv={stats['cv']:.4f} "
            f"first={stats['first']:.4f} last={stats['last']:.4f} converged={stats['converged']}"
        )
        print(
            f"  prototype={result['prototype']['resolved_by_label']} "
            f"count={result['prototype']['production_count']} "
            f"avg_sim={result['prototype']['avg_similarity']:.4f}"
        )
        results.append(result)
        print()

    passed = all(r["stats"]["cv"] <= args.cv_threshold for r in results)
    avg_cv = float(np.mean([r["stats"]["cv"] for r in results]))
    mean_similarity = float(np.mean([r["stats"]["mean"] for r in results]))

    output = {
        "config": {
            "targets": args.targets,
            "trials": args.trials,
            "duration": args.duration,
            "sr": args.sr,
            "hidden_dim": args.hidden_dim,
            "cv_threshold": args.cv_threshold,
        },
        "results": results,
        "summary": {
            "target_count": len(targets),
            "avg_cv": avg_cv,
            "mean_similarity": mean_similarity,
            "passed": passed,
            "graph_nodes": len(graph.nodes()),
            "graph_edges": len(graph.edges()),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Avg CV:          {avg_cv:.4f}")
    print(f"  Mean similarity: {mean_similarity:.4f}")
    print(f"  Graph nodes:     {len(graph.nodes())}")
    print(f"  Graph edges:     {len(graph.edges())}")
    print(f"  Verdict:         {'PASS' if passed else 'FAIL'}")
    print(f"  Saved to:        {args.output}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
