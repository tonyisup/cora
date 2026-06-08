#!/usr/bin/env python3
"""M9 temporal trainer that actually evaluates the current weights.

It hill-climbs over the controller weights loaded from the migrated producer
checkpoint, then scores each candidate locally via the current producer /
substrate / controller stack instead of calling the disk-backed smoke helper.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packages"))

import numpy as np

from m9_sound.controller import NeuralController
from m9_sound.producer import AcousticProducer
from m9_sound.temporal import (
    TEMPORAL_FAMILIES,
    apply_device_profile,
    count_decoded_transitions,
    decode_temporal_sequence,
    generate_tone_sequence,
    render_param_sequence,
    timing_alignment_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 temporal trainer")
    parser.add_argument("--family", default="default", choices=list(TEMPORAL_FAMILIES))
    parser.add_argument(
        "--base-ckpt",
        default=PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json",
        type=Path,
        help="Producer-format checkpoint with controller + substrate",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "data" / "m9_controller_temporal_trained.ckpt.json",
        type=Path,
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--noise-rms", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--device-profile", default="flat", choices=["flat", "highpass"])
    parser.add_argument("--device-gain", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--perturb-scale", type=float, default=0.12)
    parser.add_argument("--max-weight-norm", type=float, default=5.0)
    parser.add_argument("--log-step", type=int, default=1)
    parser.add_argument(
        "--heldout-family",
        default="diphthong",
        choices=list(TEMPORAL_FAMILIES),
        help="Optional held-out family used to measure generalization",
    )
    parser.add_argument(
        "--heldout-weight",
        default="auto",
        help="Blend weight for the held-out family; use 'auto' to derive it automatically",
    )
    parser.add_argument(
        "--heldout-weight-source",
        default="sweep",
        choices=["calibration", "sweep"],
        help="Choose whether auto held-out weights come from the startup calibration pair or the full family sweep summary",
    )
    return parser.parse_args()


def load_producer_ckpt(path: Path) -> AcousticProducer:
    data = json.loads(path.read_text(encoding="utf-8"))
    controller_data = data["controller"]
    substrate_data = data["substrate"]
    config = data.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 32))
    frame_size = int(config.get("frame_size", 480))
    controller_hidden_dim = int(np.asarray(controller_data["W1"]).shape[0])
    producer = AcousticProducer(sr=int(config.get("sr", 16000)), hidden_dim=hidden_dim, frame_size=frame_size)
    producer.substrate._W_h = np.asarray(substrate_data["W_h"], dtype=np.float32)
    producer.substrate._W_x = np.asarray(substrate_data["W_x"], dtype=np.float32)
    producer.substrate._b = np.asarray(substrate_data["b"], dtype=np.float32)
    producer.controller = NeuralController(hidden_dim=hidden_dim, controller_hidden_dim=controller_hidden_dim, sr=int(config.get("sr", 16000)))
    producer.controller.set_weights({k: np.asarray(v, dtype=np.float32) for k, v in controller_data.items()})
    return producer


def clip_to_norm(weights: dict[str, np.ndarray], max_norm: float) -> dict[str, np.ndarray]:
    clipped: dict[str, np.ndarray] = {}
    for key, value in weights.items():
        norm = float(np.linalg.norm(value))
        clipped[key] = value if norm <= max_norm or max_norm <= 0 else value * (max_norm / norm)
    return clipped


def derive_heldout_weight(
    primary_reward: float,
    heldout_reward: float,
    *,
    base_weight: float = 0.2,
    gap_scale: float = 1.0,
    min_weight: float = 0.15,
    max_weight: float = 0.45,
) -> float:
    """Derive a held-out blend weight from the measured family gap."""
    gap = abs(float(primary_reward) - float(heldout_reward))
    return float(np.clip(base_weight + gap_scale * gap, min_weight, max_weight))


def resolve_heldout_weight(
    weight_value: str,
    *,
    calibration_primary_reward: float,
    calibration_heldout_reward: float,
    sweep_summary: dict[str, object] | None = None,
    weight_source: str = "sweep",
) -> float:
    if weight_value != "auto":
        return float(weight_value)

    calibration_weight = derive_heldout_weight(calibration_primary_reward, calibration_heldout_reward)
    if weight_source == "sweep" and sweep_summary is not None and "mean_heldout_weight" in sweep_summary:
        return float(np.clip(float(sweep_summary["mean_heldout_weight"]), 0.15, 0.45))
    return calibration_weight


def summarize_family_pair_weights(
    producer: AcousticProducer,
    *,
    families: list[str],
    seed: int,
    noise_rms: float,
    device_profile: str,
    device_gain: float,
) -> dict[str, object]:
    primary_rewards: list[float] = []
    heldout_rewards: list[float] = []
    heldout_weights: list[float] = []
    for primary_index, primary_family in enumerate(families):
        for heldout_index, heldout_family in enumerate(families):
            pair_seed = seed + primary_index * 100 + heldout_index
            primary_reward, _ = evaluate_candidate(
                producer,
                tones=TEMPORAL_FAMILIES[primary_family],
                seed=pair_seed,
                noise_rms=noise_rms,
                device_profile=device_profile,
                device_gain=device_gain,
            )
            heldout_reward, _ = evaluate_candidate(
                producer,
                tones=TEMPORAL_FAMILIES[heldout_family],
                seed=pair_seed + 1000,
                noise_rms=noise_rms,
                device_profile=device_profile,
                device_gain=device_gain,
            )
            primary_rewards.append(float(primary_reward))
            heldout_rewards.append(float(heldout_reward))
            heldout_weights.append(float(derive_heldout_weight(primary_reward, heldout_reward)))

    if not heldout_weights:
        return {
            "mean_primary_reward": 0.0,
            "mean_heldout_reward": 0.0,
            "mean_heldout_weight": 0.0,
        }

    return {
        "mean_primary_reward": float(np.mean(np.asarray(primary_rewards, dtype=np.float32))),
        "mean_heldout_reward": float(np.mean(np.asarray(heldout_rewards, dtype=np.float32))),
        "mean_heldout_weight": float(np.mean(np.asarray(heldout_weights, dtype=np.float32))),
    }
def evaluate_candidate(
    producer: AcousticProducer,
    *,
    tones: list[tuple[float, float]],
    seed: int,
    noise_rms: float,
    device_profile: str,
    device_gain: float,
) -> tuple[float, dict[str, object]]:
    stimulus = generate_tone_sequence(tones, noise_rms=noise_rms, noise_seed=seed)
    stimulus = apply_device_profile(stimulus, device_profile=device_profile, device_gain=device_gain)
    producer.substrate.reset()
    params = decode_temporal_sequence(producer, stimulus, frame_size=producer._frame_size)
    rendered, render_report = render_param_sequence(params, sr=producer._sr, frame_size=producer._frame_size)

    target_transitions = max(0, len(tones) - 1)
    decoded_transitions = count_decoded_transitions(params, smooth_window=5 if device_profile != "flat" else 3)
    expected_order = [(freq - 40.0, freq + 40.0, freq) for freq, _ in tones[1:]]
    alignment = timing_alignment_score(
        params,
        sr=producer._sr,
        frame_size=producer._frame_size,
        expected_transitions=target_transitions,
        expected_order=expected_order,
        epsilon_hz=40.0,
    )
    transition_alignment = float(alignment["transition_recall"])
    continuity = max(0.0, 1.0 - float(render_report["mean_boundary_delta"]) * 10.0)
    timing_score = max(0.0, 1.0 - float(alignment["timing_error_frames"]))
    reward = float(0.55 * transition_alignment + 0.25 * continuity + 0.20 * timing_score)
    if noise_rms > 0.0:
        reward -= noise_rms * 10.0

    report = {
        "target_transitions": target_transitions,
        "decoded_transitions": int(decoded_transitions),
        "transition_alignment": float(transition_alignment),
        "timing_score": float(timing_score),
        "continuity": float(continuity),
        "mean_boundary_delta": float(render_report["mean_boundary_delta"]),
        "max_boundary_delta": float(render_report["max_boundary_delta"]),
        "decoded_freq_min": float(render_report["freq_min"]),
        "decoded_freq_max": float(render_report["freq_max"]),
        "rendered_samples": int(len(rendered)),
        "timing_error_frames": float(alignment["timing_error_frames"]),
        "order_coverage": float(alignment["order_coverage"]),
    }
    return reward, report


def main() -> None:
    args = parse_args()
    tones = list(TEMPORAL_FAMILIES[args.family])
    heldout_tones = None
    if args.heldout_family != args.family:
        heldout_tones = list(TEMPORAL_FAMILIES[args.heldout_family])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reward_log = args.output.with_suffix(".rewards.jsonl")
    progress_path = args.output.with_suffix(".progress.json")

    producer = load_producer_ckpt(args.base_ckpt)
    best_weights = copy.deepcopy(producer.controller.get_weights())
    best_reward: float | None = None
    last_reward = 0.0

    calibration_primary_reward = 0.0
    calibration_heldout_reward = 0.0
    sweep_summary: dict[str, object] | None = None
    sweep_mean_heldout_weight = 0.0
    resolved_heldout_weight = 0.0
    if args.heldout_family != args.family:
        producer.controller.set_weights(best_weights)
        calibration_primary_reward, _ = evaluate_candidate(
            producer,
            tones=tones,
            seed=args.noise_seed,
            noise_rms=args.noise_rms,
            device_profile=args.device_profile,
            device_gain=args.device_gain,
        )
        calibration_heldout_reward, _ = evaluate_candidate(
            producer,
            tones=heldout_tones if heldout_tones is not None else tones,
            seed=args.noise_seed + 1000,
            noise_rms=args.noise_rms,
            device_profile=args.device_profile,
            device_gain=args.device_gain,
        )
        if args.heldout_weight_source == "sweep":
            sweep_summary = summarize_family_pair_weights(
                producer,
                families=list(TEMPORAL_FAMILIES),
                seed=args.noise_seed,
                noise_rms=args.noise_rms,
                device_profile=args.device_profile,
                device_gain=args.device_gain,
            )
            sweep_mean_heldout_weight = float(cast(float, sweep_summary["mean_heldout_weight"]))
        resolved_heldout_weight = resolve_heldout_weight(
            args.heldout_weight,
            calibration_primary_reward=calibration_primary_reward,
            calibration_heldout_reward=calibration_heldout_reward,
            sweep_summary=sweep_summary,
            weight_source=args.heldout_weight_source,
        )
    else:
        resolved_heldout_weight = float(args.heldout_weight) if args.heldout_weight != "auto" else 0.0

    with reward_log.open("a", encoding="utf-8") as fh:
        for episode in range(1, args.episodes + 1):
            candidate_weights = copy.deepcopy(best_weights)
            if episode > 1:
                step = args.lr * float(np.tanh(last_reward + 1e-9))
                for key, value in candidate_weights.items():
                    noise = np.random.randn(*value.shape).astype(np.float32) * args.perturb_scale
                    candidate_weights[key] = value + noise * step
                candidate_weights = clip_to_norm(candidate_weights, args.max_weight_norm)

            producer.controller.set_weights(candidate_weights)
            seed = args.noise_seed + episode
            reward, report = evaluate_candidate(
                producer,
                tones=tones,
                seed=seed,
                noise_rms=args.noise_rms,
                device_profile=args.device_profile,
                device_gain=args.device_gain,
            )
            heldout_reward = None
            heldout_report = None
            if heldout_tones is not None:
                heldout_reward, heldout_report = evaluate_candidate(
                    producer,
                    tones=heldout_tones,
                    seed=seed + 1000,
                    noise_rms=args.noise_rms,
                    device_profile=args.device_profile,
                    device_gain=args.device_gain,
                )
                reward = float((1.0 - resolved_heldout_weight) * reward + resolved_heldout_weight * heldout_reward)

            updated = False
            if best_reward is None or reward > best_reward:
                best_reward = reward
                best_weights = copy.deepcopy(candidate_weights)
                updated = True

            fh.write(
                json.dumps(
                    {
                        "episode": episode,
                        "family": args.family,
                        "seed": seed,
                        "reward": reward,
                        "updated": updated,
                        **report,
                        **({"heldout_reward": heldout_reward, **heldout_report} if heldout_report is not None else {}),
                        "resolved_heldout_weight": resolved_heldout_weight,
                        "heldout_weight_source": args.heldout_weight_source,
                        "sweep_mean_heldout_weight": sweep_mean_heldout_weight,
                        "calibration_primary_reward": calibration_primary_reward,
                        "calibration_heldout_reward": calibration_heldout_reward,
                    }
                )
                + "\n"
            )
            fh.flush()

            progress_path.write_text(
                json.dumps(
                    {
                        "episode": episode,
                        "best_reward": best_reward,
                        "last_reward": reward,
                        "family": args.family,
                        "updated": updated,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if episode % args.log_step == 0 or episode == 1:
                heldout_str = ""
                if heldout_report is not None:
                    heldout_str = (
                        f" heldout_reward={heldout_reward:+.3f} heldout_decoded={heldout_report['decoded_transitions']}"
                        f" heldout_weight={resolved_heldout_weight:+.3f} heldout_source={args.heldout_weight_source}"
                    )
                print(
                    f"episode {episode:03d} reward={reward:+.3f} best={best_reward:+.3f} "
                    f"decoded={report['decoded_transitions']} updated={updated}{heldout_str}"
                )

            last_reward = reward

    args.output.write_text(
        json.dumps({k: v.tolist() for k, v in best_weights.items()}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_reward": best_reward,
                "output": str(args.output),
                "resolved_heldout_weight": resolved_heldout_weight,
                "heldout_weight_source": args.heldout_weight_source,
                "sweep_mean_heldout_weight": sweep_mean_heldout_weight,
                "calibration_primary_reward": calibration_primary_reward,
                "calibration_heldout_reward": calibration_heldout_reward,
            }
        )
    )


if __name__ == "__main__":
    main()
