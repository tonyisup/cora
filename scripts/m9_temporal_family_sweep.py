#!/usr/bin/env python3
"""Sweep M9 temporal family pairs and report generalization across them.

This is a thin CLI around the local trainer evaluation helpers. It keeps the
scoring path in-memory and reports primary/held-out rewards for every ordered
family pair.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packages"))


def load_trainer_module():
    trainer_path = PROJECT_ROOT / "scripts" / "m9_temporal_train.py"
    spec = importlib.util.spec_from_file_location("m9_temporal_train", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load trainer module from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 temporal family pair sweep")
    parser.add_argument(
        "--base-ckpt",
        default=PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json",
        type=Path,
        help="Producer-format checkpoint with controller + substrate",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=list(TRAINER.TEMPORAL_FAMILIES),
        choices=list(TRAINER.TEMPORAL_FAMILIES),
        help="Temporal families to sweep in ordered pairs",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--noise-rms", type=float, default=0.0)
    parser.add_argument("--device-profile", default="flat", choices=["flat", "highpass"])
    parser.add_argument("--device-gain", type=float, default=1.0)
    parser.add_argument(
        "--heldout-weight",
        default="auto",
        help="Blend weight for the held-out family in the pair score; use 'auto' to derive it from the measured family gap",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "data" / "m9_temporal_family_pair_sweep.json",
        type=Path,
    )
    return parser.parse_args()


def evaluate_family_pair(
    producer,
    *,
    primary_family: str,
    heldout_family: str,
    seed: int,
    noise_rms: float,
    device_profile: str,
    device_gain: float,
    heldout_weight: str | float,
) -> dict[str, object]:
    primary_reward, primary_report = TRAINER.evaluate_candidate(
        producer,
        tones=TRAINER.TEMPORAL_FAMILIES[primary_family],
        seed=seed,
        noise_rms=noise_rms,
        device_profile=device_profile,
        device_gain=device_gain,
    )
    heldout_reward, heldout_report = TRAINER.evaluate_candidate(
        producer,
        tones=TRAINER.TEMPORAL_FAMILIES[heldout_family],
        seed=seed + 1000,
        noise_rms=noise_rms,
        device_profile=device_profile,
        device_gain=device_gain,
    )
    resolved_heldout_weight = (
        TRAINER.derive_heldout_weight(primary_reward, heldout_reward)
        if heldout_weight == "auto"
        else float(heldout_weight)
    )
    blended_reward = float((1.0 - resolved_heldout_weight) * primary_reward + resolved_heldout_weight * heldout_reward)

    return {
        "primary_family": primary_family,
        "heldout_family": heldout_family,
        "seed": seed,
        "primary_reward": float(primary_reward),
        "heldout_reward": float(heldout_reward),
        "blended_reward": float(blended_reward),
        "heldout_weight": float(resolved_heldout_weight),
        "primary_decoded": int(primary_report["decoded_transitions"]),
        "heldout_decoded": int(heldout_report["decoded_transitions"]),
        "primary_transition_alignment": float(primary_report["transition_alignment"]),
        "heldout_transition_alignment": float(heldout_report["transition_alignment"]),
        "primary_timing_score": float(primary_report["timing_score"]),
        "heldout_timing_score": float(heldout_report["timing_score"]),
        "primary_report": primary_report,
        "heldout_report": heldout_report,
    }


def build_family_pair_sweep_report(
    *,
    base_ckpt: str | Path,
    families: Iterable[str] | None = None,
    seed: int = 43,
    noise_rms: float = 0.0,
    device_profile: str = "flat",
    device_gain: float = 1.0,
    heldout_weight: float = 0.25,
) -> dict[str, object]:
    families_list = list(TRAINER.TEMPORAL_FAMILIES if families is None else families)
    producer = TRAINER.load_producer_ckpt(Path(base_ckpt))

    pairs: list[dict[str, object]] = []
    for primary_index, primary_family in enumerate(families_list):
        for heldout_index, heldout_family in enumerate(families_list):
            pair_seed = seed + primary_index * 100 + heldout_index
            pairs.append(
                evaluate_family_pair(
                    producer,
                    primary_family=primary_family,
                    heldout_family=heldout_family,
                    seed=pair_seed,
                    noise_rms=noise_rms,
                    device_profile=device_profile,
                    device_gain=device_gain,
                    heldout_weight=heldout_weight,
                )
            )

    summary: dict[str, object]
    if pairs:
        blended = np.array([float(cast(float, pair["blended_reward"])) for pair in pairs], dtype=np.float32)
        primary = np.array([float(cast(float, pair["primary_reward"])) for pair in pairs], dtype=np.float32)
        heldout = np.array([float(cast(float, pair["heldout_reward"])) for pair in pairs], dtype=np.float32)
        best_idx = int(np.argmax(blended))
        best_pair = pairs[best_idx]
        summary = {
            "mean_primary_reward": float(np.mean(primary)),
            "mean_heldout_reward": float(np.mean(heldout)),
            "mean_blended_reward": float(np.mean(blended)),
            "mean_heldout_weight": float(np.mean([float(cast(float, pair["heldout_weight"])) for pair in pairs])),
            "best_pair": {
                "primary_family": cast(str, best_pair["primary_family"]),
                "heldout_family": cast(str, best_pair["heldout_family"]),
                "blended_reward": float(cast(float, best_pair["blended_reward"])),
            },
        }
    else:
        summary = {
            "mean_primary_reward": 0.0,
            "mean_heldout_reward": 0.0,
            "mean_blended_reward": 0.0,
            "best_pair": None,
        }

    return {
        "base_ckpt": str(Path(base_ckpt)),
        "families": families_list,
        "pair_count": len(pairs),
        "pairs": pairs,
        "summary": summary,
        "seed": seed,
        "noise_rms": float(noise_rms),
        "device_profile": device_profile,
        "device_gain": float(device_gain),
        "heldout_weight": heldout_weight,
    }


def main() -> None:
    args = parse_args()
    report = build_family_pair_sweep_report(
        base_ckpt=args.base_ckpt,
        families=args.families,
        seed=args.seed,
        noise_rms=args.noise_rms,
        device_profile=args.device_profile,
        device_gain=args.device_gain,
        heldout_weight=args.heldout_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pairs_out = cast(list[dict[str, object]], report["pairs"])
    for pair in pairs_out:
        print(
            f"{cast(str, pair['primary_family'])} -> {cast(str, pair['heldout_family'])} "
            f"primary={float(cast(float, pair['primary_reward'])):+.3f} heldout={float(cast(float, pair['heldout_reward'])):+.3f} "
            f"blended={float(cast(float, pair['blended_reward'])):+.3f} "
            f"weight={float(cast(float, pair['heldout_weight'])):+.3f} "
            f"decoded={cast(int, pair['primary_decoded'])}/{cast(int, pair['heldout_decoded'])}"
        )
    print(json.dumps({"output": str(args.output), "pair_count": report["pair_count"], "summary": report["summary"]}))


if __name__ == "__main__":
    main()
