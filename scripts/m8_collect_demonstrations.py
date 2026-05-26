#!/usr/bin/env python3
"""M8b — Collect heuristic demonstrations for distilled policy training.

Runs the heuristic speaking policy on scripted episodes and collects
(features, action) pairs as training data for the distilled MLP.

Usage:
    .venv/bin/python scripts/m8_collect_demonstrations.py \\
        --episodes scripts/m8_episodes_default.json \\
        --num-seeds 5 \\
        --output /tmp/m8_demonstrations.jsonl

    # Save default expanded episodes first:
    .venv/bin/python scripts/m8_collect_demonstrations.py \\
        --save-episodes scripts/m8_episodes_default.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.conversation_tracker import ConversationTracker
from pfc.speaking_policy import HeuristicSpeakingPolicy
from m8_conversation_simulator import create_default_episodes, load_episodes


def perturb_episode(episode: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Create a perturbed copy of an episode with randomized timing.

    This generates diverse training data from the same episode structure
    by jittering event times and durations.
    """
    rng = random.Random(seed)
    ep = copy.deepcopy(episode)
    ep["id"] = f"{episode['id']}_seed{seed}"

    # Jitter agent output times by ±0.5s
    for ao in ep.get("agent_output", []):
        jitter = rng.uniform(-0.5, 0.5)
        ao["time"] = max(0.0, ao["time"] + jitter)
        # Jitter duration by ±10%
        dur_jitter = rng.uniform(-0.1, 0.1)
        ao["duration"] = max(0.5, ao["duration"] * (1.0 + dur_jitter))

    # Jitter user speech times by ±0.3s
    for us in ep.get("user_speech", []):
        start_jitter = rng.uniform(-0.3, 0.3)
        us["start"] = max(0.0, us["start"] + start_jitter)
        dur_jitter = rng.uniform(-0.1, 0.1)
        dur = us["end"] - us["start"]
        us["end"] = us["start"] + max(0.3, dur * (1.0 + dur_jitter))

    return ep


def collect_from_episode(
    episode: Dict[str, Any],
    policy: HeuristicSpeakingPolicy,
) -> List[Dict[str, Any]]:
    """Run a single episode and collect (features, action) pairs."""
    duration = episode["duration_s"]
    dt = episode.get("time_step_s", 0.03)
    agent_output = episode.get("agent_output", [])
    user_speech = episode.get("user_speech", [])

    tracker = ConversationTracker()
    policy.reset()
    samples = []

    t = 0.0
    while t < duration:
        # Build frame (same logic as simulator)
        frame = _build_frame(t, agent_output, user_speech)
        tracker.process_frame(frame)
        features = tracker.get_features()

        # Set has_queued_output
        has_queued = any(t < ao["time"] + ao["duration"] for ao in agent_output)
        features["has_queued_output"] = has_queued

        action = policy.decide(features)

        samples.append({
            "features": features,
            "action": action,
            "episode_id": episode.get("id", "unknown"),
            "time_s": round(t, 3),
        })

        t += dt

    return samples


def _build_frame(
    t: float,
    agent_output: List[Dict[str, Any]],
    user_speech: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a synthetic frame at time t (simplified from simulator)."""
    tts_playing = False
    for ao in agent_output:
        if ao["time"] <= t < ao["time"] + ao["duration"]:
            tts_playing = True
            break

    user_speaking = any(us["start"] <= t < us["end"] for us in user_speech)

    if tts_playing and user_speaking:
        verdict = "BOTH"
    elif tts_playing:
        verdict = "SELF"
    elif user_speaking:
        verdict = "WORLD"
    else:
        verdict = "SILENCE"

    if tts_playing:
        controller_state = "SPEAK"
    elif user_speaking:
        controller_state = "LISTEN"
    else:
        controller_state = "IDLE"

    return {
        "frame_index": int(t / 0.03),
        "timestamp_s": t,
        "vad_speech": verdict != "SILENCE",
        "verdict": verdict,
        "controller_state": controller_state,
        "can_speak": controller_state in ("IDLE", "LISTEN"),
        "should_interrupt": verdict == "BOTH" and tts_playing,
        "tts_playing": tts_playing,
        "tts_position_s": 0.0,
        "tts_total_duration_s": 3.0,
        "output_source": "tts" if tts_playing else "silence",
        "repair_position_s": 0.0,
        "repair_total_duration_s": 0.0,
        "repair_queued": False,
        "repair_playing": False,
        "repair_completed": False,
        "m4_outcome": "not_attempted",
        "m4_confidence": None,
        "energy_db": -30.0 if verdict != "SILENCE" else -60.0,
        "self_match_score": 0.0,
        "self_match_score_per_frame": None,
        "is_in_motor_window": tts_playing,
        "deferred": False,
        "resumed": False,
        "collision_repair_text": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M8b collect heuristic demonstrations")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of perturbed variants per episode")
    parser.add_argument("--output", type=str, default="/tmp/m8_demonstrations.jsonl")
    parser.add_argument("--save-episodes", type=str, default=None,
                        help="Save default episodes to file and exit")
    args = parser.parse_args()

    if args.save_episodes:
        episodes = create_default_episodes()
        with open(args.save_episodes, "w") as f:
            json.dump({"episodes": episodes}, f, indent=2)
        print(f"Saved {len(episodes)} episodes to {args.save_episodes}")
        return

    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        episodes = create_default_episodes()

    policy = HeuristicSpeakingPolicy()

    all_samples = []
    print(f"Collecting demonstrations from {len(episodes)} episodes x {args.num_seeds} seeds...")

    for ep in episodes:
        # Collect from the base episode
        samples = collect_from_episode(ep, policy)
        all_samples.extend(samples)
        print(f"  {ep['id']:30s}  {len(samples)} samples")

        # Collect from perturbed variants
        for seed in range(args.num_seeds):
            perturbed = perturb_episode(ep, seed)
            samples = collect_from_episode(perturbed, policy)
            all_samples.extend(samples)
            print(f"  {perturbed['id']:30s}  {len(samples)} samples (perturbed)")

    print(f"\nTotal samples collected: {len(all_samples)}")

    # Action distribution
    from collections import Counter
    action_counts = Counter(s["action"] for s in all_samples)
    print(f"Action distribution: {dict(action_counts)}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
