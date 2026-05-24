#!/usr/bin/env python3
"""M8 — Offline conversation simulator for speaking-policy evaluation.

Runs scripted conversational episodes through a SpeakingPolicy and
scores the results with the M8 reward function.

An episode is a sequence of timed events that simulate a conversation:
- user speech start/stop
- agent output queued
- user interruptions
- user corrections

The simulator feeds events into the ConversationTracker → SpeakingPolicy → Reward
pipeline and produces a step-by-step trace for analysis.

Usage:
    .venv/bin/python scripts/m8_conversation_simulator.py \\
        --episodes scripts/m8_episodes_default.json \\
        --policy heuristic \\
        --output /tmp/m8_sim_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.conversation_tracker import ConversationTracker
from pfc.speaking_policy import (
    HeuristicSpeakingPolicy,
    SpeakingPolicy,
    ACTION_NAMES,
)
from m8_reward import M8RewardConfig, compute_m8_reward


# ── Episode format ──────────────────────────────────────────────────────────

def load_episodes(path: str) -> List[Dict[str, Any]]:
    """Load scripted episodes from a JSON file.

    Expected format:
    {
      "episodes": [
        {
          "id": "episode_001",
          "description": "Agent speaks, user interrupts, agent yields",
          "duration_s": 10.0,
          "time_step_s": 0.03,
          "agent_output": [
            {"time": 1.0, "duration": 3.0, "text": "Hello, I'm Cora."}
          ],
          "user_speech": [
            {"start": 0.0, "end": 0.5},
            {"start": 2.0, "end": 2.5}
          ]
        }
      ]
    }
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["episodes"]


def create_default_episodes() -> List[Dict[str, Any]]:
    """Create a default set of scripted episodes for M8 evaluation."""
    return [
        {
            "id": "speak_on_silence",
            "description": "Agent speaks after period of silence",
            "duration_s": 5.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 1.0, "duration": 3.0, "text": "Hello, I'm Cora."},
            ],
            "user_speech": [],
        },
        {
            "id": "yield_on_interrupt",
            "description": "Agent is speaking, user interrupts, agent yields",
            "duration_s": 8.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 1.0, "duration": 5.0, "text": "I was thinking about what you said."},
            ],
            "user_speech": [
                {"start": 2.5, "end": 4.0},
            ],
        },
        {
            "id": "resume_after_yield",
            "description": "After yielding, agent resumes when user finishes",
            "duration_s": 10.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 1.0, "duration": 6.0, "text": "Let me tell you about my day."},
            ],
            "user_speech": [
                {"start": 2.0, "end": 3.5},
                {"start": 7.0, "end": 8.0},
            ],
        },
        {
            "id": "wait_while_user_speaks",
            "description": "Agent waits while user is speaking",
            "duration_s": 6.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 0.5, "duration": 4.0, "text": "I'd like to ask you something."},
            ],
            "user_speech": [
                {"start": 0.0, "end": 4.0},
            ],
        },
        {
            "id": "repair_after_collision",
            "description": "Agent and user speak simultaneously, agent repairs",
            "duration_s": 8.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 1.0, "duration": 4.0, "text": "I wanted to mention something."},
            ],
            "user_speech": [
                {"start": 2.5, "end": 5.0},
            ],
        },
        {
            "id": "user_silence_then_speak",
            "description": "User speaks, then silence, agent speaks",
            "duration_s": 8.0,
            "time_step_s": 0.03,
            "agent_output": [
                {"time": 0.5, "duration": 5.0, "text": "Can I tell you my idea?"},
            ],
            "user_speech": [
                {"start": 0.0, "end": 2.0},
            ],
        },
    ]


# ── Simulator ───────────────────────────────────────────────────────────────

def build_frame_at_time(
    t: float,
    agent_output: List[Dict[str, Any]],
    user_speech: List[Dict[str, Any]],
    frame_index: int,
    prev_features: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a synthetic frame log entry at time t.

    Simulates what the streaming runtime's frame log would look like
    at a given point in the episode.
    """
    # Determine TTS state
    tts_playing = False
    tts_position_s = 0.0
    tts_total_duration_s = 0.0
    output_started = False
    for ao in agent_output:
        ao_start = ao["time"]
        ao_end = ao["time"] + ao["duration"]
        if ao_start <= t:
            output_started = True
            tts_total_duration_s = ao["duration"]
        if ao_start <= t < ao_end:
            tts_playing = True
            tts_position_s = t - ao_start
            break

    # Determine user speech state
    user_speaking = False
    for us in user_speech:
        if us["start"] <= t < us["end"]:
            user_speaking = True
            break

    # Determine verdict
    if tts_playing and user_speaking:
        verdict = "BOTH"
    elif tts_playing:
        verdict = "SELF"
    elif user_speaking:
        verdict = "WORLD"
    else:
        verdict = "SILENCE"

    # Determine controller state (simplified — follows agent activity)
    if tts_playing:
        controller_state = "SPEAK"
    elif user_speaking:
        controller_state = "LISTEN"
    else:
        controller_state = "IDLE"

    # Detect collision (BOTH while agent is speaking)
    just_deferred = False
    collision_repair_text = None
    if verdict == "BOTH" and prev_features.get("is_agent_speaking", False):
        just_deferred = True
        collision_repair_text = "Sorry, go ahead."

    # Detect resume (agent was deferred, now speaking again)
    just_resumed = (
        tts_playing and
        prev_features.get("is_agent_deferred", False) and
        not just_deferred
    )

    return {
        "frame_index": frame_index,
        "timestamp_s": t,
        "vad_speech": verdict != "SILENCE",
        "verdict": verdict,
        "controller_state": controller_state,
        "can_speak": controller_state in ("IDLE", "LISTEN"),
        "should_interrupt": verdict == "BOTH",
        "tts_playing": tts_playing,
        "tts_position_s": tts_position_s,
        "tts_total_duration_s": tts_total_duration_s,
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
        "deferred": just_deferred,
        "resumed": just_resumed,
        "collision_repair_text": collision_repair_text,
    }


def run_episode(
    episode: Dict[str, Any],
    policy: SpeakingPolicy,
    reward_config: M8RewardConfig,
) -> Dict[str, Any]:
    """Run a single episode through the simulator.

    Returns a dict with episode id, per-step trace, total reward, and summary.
    """
    duration = episode["duration_s"]
    dt = episode.get("time_step_s", 0.03)
    agent_output = episode.get("agent_output", [])
    user_speech = episode.get("user_speech", [])

    tracker = ConversationTracker()
    policy.reset()

    steps = []
    total_reward = 0.0
    action_counts: Dict[str, int] = {a: 0 for a in ACTION_NAMES}
    reward_components: Dict[str, float] = {}

    prev_features: Dict[str, Any] = tracker.get_features()
    frame_index = 0

    t = 0.0
    while t < duration:
        # Build synthetic frame
        frame = build_frame_at_time(t, agent_output, user_speech, frame_index, prev_features)

        # Update tracker
        tracker.process_frame(frame)

        # Get features for policy
        features = tracker.get_features()

        # Set has_queued_output based on whether agent has pending output
        has_queued = False
        for ao in agent_output:
            ao_end = ao["time"] + ao["duration"]
            if t < ao_end:
                has_queued = True
                break
        features["has_queued_output"] = has_queued

        # Policy decides
        action = policy.decide(features)

        # Compute reward (features after action → next_features)
        next_t = t + dt
        next_frame = build_frame_at_time(next_t, agent_output, user_speech, frame_index + 1, features)
        # We need to process the next frame through the tracker to get next_features
        # but we don't want to double-count. Instead, we'll use the current features
        # as a proxy for the "next" state reward evaluation.
        reward_signal = compute_m8_reward(reward_config, action, prev_features, features)

        total_reward += reward_signal.reward
        action_counts[action] = action_counts.get(action, 0) + 1
        for k, v in reward_signal.components.items():
            reward_components[k] = reward_components.get(k, 0.0) + v

        steps.append({
            "time_s": round(t, 3),
            "action": action,
            "reward": round(reward_signal.reward, 4),
            "reward_reason": reward_signal.reason,
            "verdict": frame["verdict"],
            "controller_state": frame["controller_state"],
            "conversation_state": features.get("conversation_state", "unknown"),
        })

        prev_features = features.copy()
        frame_index += 1
        t += dt

    # Summary
    total_steps = len(steps)
    non_wait_steps = sum(1 for s in steps if s["action"] != "wait")

    return {
        "episode_id": episode["id"],
        "description": episode.get("description", ""),
        "policy": policy.__class__.__name__,
        "total_reward": round(total_reward, 4),
        "total_steps": total_steps,
        "action_counts": action_counts,
        "reward_components": {k: round(v, 4) for k, v in reward_components.items()},
        "non_wait_ratio": round(non_wait_steps / max(total_steps, 1), 4),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 conversation simulator")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Path to JSON episode file (default: built-in episodes)")
    parser.add_argument("--policy", type=str, default="heuristic",
                        choices=["heuristic"],
                        help="Policy to evaluate")
    parser.add_argument("--output", type=str, default="/tmp/m8_sim_results.jsonl",
                        help="Output path for results JSONL")
    parser.add_argument("--save-episodes", type=str, default=None,
                        help="Save default episodes to a JSON file and exit")
    args = parser.parse_args()

    if args.save_episodes:
        episodes = create_default_episodes()
        with open(args.save_episodes, "w") as f:
            json.dump({"episodes": episodes}, f, indent=2)
        print(f"Saved {len(episodes)} default episodes to {args.save_episodes}")
        return

    # Load episodes
    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        episodes = create_default_episodes()

    # Create policy
    if args.policy == "heuristic":
        policy: SpeakingPolicy = HeuristicSpeakingPolicy()
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    reward_config = M8RewardConfig()

    # Run all episodes
    print(f"Running {len(episodes)} episodes with {policy.__class__.__name__}...")
    print("-" * 60)

    all_results = []
    total_reward = 0.0

    for ep in episodes:
        result = run_episode(ep, policy, reward_config)
        all_results.append(result)
        total_reward += result["total_reward"]
        print(f"  {result['episode_id']:30s}  reward={result['total_reward']:8.4f}  "
              f"actions={result['action_counts']}")

    print("-" * 60)
    print(f"  Total reward across all episodes: {total_reward:.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
