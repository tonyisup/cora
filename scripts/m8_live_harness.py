#!/usr/bin/env python3
"""M8 Live Hardware Validation Harness.

Runs scripted conversational episodes on real audio hardware with the M8
speaking policy making real-time decisions. Validates that offline policy
transfers to live audio with real acoustic latency and noise.

Episode protocol:
1. Agent speaks a TTS utterance through the primary speaker
2. At a scripted time, a "human" voice (TTS on 2nd device) starts
3. The streaming runtime detects collision and gates off agent TTS
4. The speaking policy reads frame logs and decides: yield/wait/repair/speak
5. Log all decisions + timestamps for post-hoc analysis

Usage:
    # List available devices
    .venv/bin/python scripts/m8_live_harness.py --list-devices

    # Run with hardware (requires mic + 2 output devices)
    .venv/bin/python scripts/m8_live_harness.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --human-output-device 2 \\
        --policy distilled \\
        --episodes all

    # Run specific episode
    .venv/bin/python scripts/m8_live_harness.py \\
        --input-device 1 \\
        --self-output-device 3 \\
        --human-output-device 2 \\
        --policy ppo \\
        --episodes speak_on_silence

    # Dry run (no audio devices needed — validates episode scripts)
    .venv/bin/python scripts/m8_live_harness.py --dry-run --episodes all

Requirements:
    - Mic connected
    - Primary speaker for agent TTS
    - Second audio source for human voice (second speaker, phone, laptop)
    - Fine-tuned M4 scorer for self/world attribution
    - Room filter for current acoustic environment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from pfc.conversation_tracker import ConversationTracker
from pfc.speaking_policy import HeuristicSpeakingPolicy, SpeakingPolicy, ACTION_NAMES
from pfc.streaming_runtime import StreamingConfig, StreamingRuntime
from broca.tts import synthesize as _synthesize_tts


# ── Episode Definitions ─────────────────────────────────────────────────────

LIVE_EPISODES = [
    {
        "id": "speak_on_silence",
        "description": "Agent speaks after silence, no interruption",
        "duration_s": 6.0,
        "agent_utterances": [
            {"time": 0.0, "voice": "en-US-AriaNeural", "text": "Hello, I am Cora. Testing the speaking policy on live hardware."},
        ],
        "user_voice": [],
        "expected_actions": ["speak"],
    },
    {
        "id": "yield_on_interrupt",
        "description": "Agent speaks, human interrupts, agent must yield",
        "duration_s": 8.0,
        "agent_utterances": [
            {"time": 0.0, "voice": "en-US-AriaNeural", "text": "I am speaking a long utterance to test interruption handling."},
            {"time": 5.0, "voice": "en-US-AriaNeural", "text": "And now I am resuming after the interruption."},
        ],
        "user_voice": [
            {"start": 2.0, "voice": "en-US-GuyNeural", "text": "Excuse me, I need to say something important."},
        ],
        "expected_actions": ["speak", "yield", "speak"],
    },
    {
        "id": "wait_while_user_speaks",
        "description": "User speaks, agent has queued output but must wait",
        "duration_s": 6.0,
        "agent_utterances": [
            {"time": 0.0, "voice": "en-US-AriaNeural", "text": "I would like to respond to what you just said. Let me think about it."},
        ],
        "user_voice": [
            {"start": 0.5, "voice": "en-US-GuyNeural", "text": "I have been thinking about this topic for a long time and I have some thoughts I want to share with you."},
        ],
        "expected_actions": ["wait", "speak"],
    },
]


# ── TTS Generation ──────────────────────────────────────────────────────────

SR = 16000


def make_tts_pcm(text: str, voice: str = "en-US-AriaNeural") -> np.ndarray:
    """Generate TTS audio."""
    pcm = asyncio.run(_synthesize_tts(text, voice=voice))
    return pcm.astype(np.float32)


def list_devices() -> None:
    """Print available audio devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Cannot list devices.")
        return
    print("Available audio devices:")
    print(f"  {'#':>3}  {'Name':<30}  {'In':>3}  {'Out':>3}")
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 or d['max_output_channels'] > 0:
            print(f"  {i:3d}  {d['name']:<30}  {d['max_input_channels']:3d}  {d['max_max_output_channels']:3d}" if 'max_max_output_channels' in d else f"  {i:3d}  {d['name']:<30}  {d['max_input_channels']:3d}  {d['max_output_channels']:3d}")
    print()
    print("Usage:")
    print("  --input-device 1 --self-output-device 3 --human-output-device 2")


# ── Human Voice Playback ────────────────────────────────────────────────────

def play_voice_on_device(
    pcm: np.ndarray,
    device: int,
    delay_s: float,
    start_reference: float,
) -> None:
    """Play audio on a device after a delay. Blocking."""
    import sounddevice as sd
    actual_start = start_reference + delay_s
    wait = actual_start - time.time()
    if wait > 0:
        time.sleep(wait)
    sd.play(pcm, samplerate=SR, device=device)
    sd.wait()


# ── Policy Loading ──────────────────────────────────────────────────────────

def load_speaking_policy(policy_type: str, checkpoint_path: Optional[str] = None) -> SpeakingPolicy:
    """Load a speaking policy by type."""
    if policy_type == "heuristic":
        return HeuristicSpeakingPolicy()
    elif policy_type == "distilled":
        from m8_policy.model import M8DistilledPolicy, encode_m8_features

        device = "cpu"
        ckpt = torch_load_safe(checkpoint_path or "data/m8_distilled_policy.pt")
        if ckpt is None:
            print("WARNING: Distilled checkpoint not found, using heuristic")
            return HeuristicSpeakingPolicy()
        config = ckpt.get("config", {})
        model = M8DistilledPolicy(
            input_dim=config.get("input_dim", 21),
            hidden_dim=config.get("hidden_dim", 64),
            dropout=0.0,
            num_actions=config.get("num_actions", 5),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return _DistilledPolicyWrapper(model)
    elif policy_type == "ppo":
        from m8_train_ppo import M8ActorCritic, encode_obs

        device = "cpu"
        ckpt = torch_load_safe(checkpoint_path or "results/m8_ppo_scripted_v2/best_policy.pt")
        if ckpt is None:
            print("WARNING: PPO checkpoint not found, using heuristic")
            return HeuristicSpeakingPolicy()
        config = ckpt.get("config", {})
        model = M8ActorCritic(
            input_dim=config.get("input_dim", 13),
            hidden_dim=config.get("hidden_dim", 64),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["policy_state"])
        model.eval()
        return _PPOPolicyWrapper(model)
    else:
        print(f"Unknown policy type: {policy_type}, using heuristic")
        return HeuristicSpeakingPolicy()


def torch_load_safe(path: str) -> Optional[Dict]:
    """Safely load a torch checkpoint. Returns None on failure."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        return ckpt
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None


class _DistilledPolicyWrapper(SpeakingPolicy):
    def __init__(self, model):
        self._model = model

    def decide(self, features: Dict[str, Any]) -> str:
        from m8_policy.model import encode_m8_features
        import torch
        x = encode_m8_features(features).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
            idx = logits.argmax(dim=-1).item()
        return ACTION_NAMES[idx]


class _PPOPolicyWrapper(SpeakingPolicy):
    def __init__(self, model):
        self._model = model

    def decide(self, features: Dict[str, Any]) -> str:
        from m8_train_ppo import encode_obs
        import torch
        # Convert conversation features to obs format
        obs = _conv_to_obs(features)
        x = encode_obs(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._model(x)
            idx = logits.argmax(dim=-1).item()
        return ACTION_NAMES[idx]


def _conv_to_obs(features: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ConversationFeatures to streaming runtime obs format."""
    state_map = {"idle": "idle", "listening": "listen", "speaking": "speak",
                 "repair": "repair", "clarification": "clarify"}
    return {
        "state": state_map.get(features.get("conversation_state", "listen"), "listen"),
        "attribution": "silence",
        "silence_duration_ms": features.get("seconds_since_last_user_speech", 0.0) * 1000.0,
        "has_queued_utterance": features.get("has_queued_output", False),
        "user_speaking": features.get("is_user_speaking", False),
        "agent_speaking": features.get("is_agent_speaking", False),
        "queue_depth": 1 if features.get("is_agent_deferred", False) else 0,
    }


# ── Episode Runner ───────────────────────────────────────────────────────────

def build_episode_script(episode: Dict) -> List[Dict[str, Any]]:
    """Build a unified timeline of events for an episode.

    Returns a sorted list of events: agent TTS, user voice, policy decision points.
    """
    events = []

    for utt in episode.get("agent_utterances", []):
        events.append({
            "time": utt["time"],
            "type": "agent_tts",
            "voice": utt.get("voice", "en-US-AriaNeural"),
            "text": utt["text"],
        })

    for uv in episode.get("user_voice", []):
        events.append({
            "time": uv["start"],
            "type": "user_voice",
            "voice": uv.get("voice", "en-US-GuyNeural"),
            "text": uv["text"],
        })

    # Policy decision points: before each agent utterance + after each user voice
    for utt in episode.get("agent_utterances", []):
        events.append({
            "time": max(0, utt["time"] - 0.5),  # 500ms before each utterance
            "type": "policy_decision",
            "context": f"before_utterance_{utt['text'][:20]}",
        })

    for uv in episode.get("user_voice", []):
        events.append({
            "time": uv["start"] + 0.1,  # 100ms after user starts
            "type": "policy_decision",
            "context": f"during_user_voice",
        })

    events.sort(key=lambda e: e["time"])
    return events


def run_episode_dry(episode: Dict, policy: SpeakingPolicy) -> Dict[str, Any]:
    """Run an episode in dry-run mode — no audio devices needed."""
    tracker = ConversationTracker()
    policy.reset()
    events = build_episode_script(episode)

    timeline = []
    tts_position = 0.0

    for event in events:
        # Simulate time progression
        if event["type"] == "agent_tts":
            # Simulate speaking
            pcm = make_tts_pcm(event["text"], event["voice"])
            duration = len(pcm) / SR
            tts_position = event["time"]

            # Log
            timeline.append({
                "time": event["time"],
                "type": "agent_tts",
                "text": event["text"],
                "duration": round(duration, 2),
            })

        elif event["type"] == "user_voice":
            timeline.append({
                "time": event["time"],
                "type": "user_voice",
                "text": event["text"],
            })

        elif event["type"] == "policy_decision":
            # Run policy on current tracker state
            features = tracker.get_features()
            action = policy.decide(features)
            timeline.append({
                "time": event["time"],
                "type": "policy_decision",
                "action": action,
                "context": event.get("context", ""),
                "features": {k: v for k, v in features.items()
                             if k.startswith("is_") or k == "conversation_state"},
            })

    return {
        "episode_id": episode["id"],
        "mode": "dry_run",
        "duration_s": episode["duration_s"],
        "timeline": timeline,
        "policy": policy.__class__.__name__,
    }


def run_episode_live(
    episode: Dict,
    policy: SpeakingPolicy,
    args,
) -> Dict[str, Any]:
    """Run an episode on live hardware.

    Requires audio devices. Uses StreamingRuntime for real audio I/O.
    Speaking policy runs in orchestrator loop between utterances.
    """
    import threading

    tracker = ConversationTracker()
    policy.reset()
    sr = SR

    # Pre-generate all TTS audio
    print(f"  Generating TTS audio...")
    tts_segments = {}
    for i, utt in enumerate(episode.get("agent_utterances", [])):
        pcm = make_tts_pcm(utt["text"], utt["voice"])
        tts_segments[("agent", i)] = pcm
        print(f"    Agent utterance {i}: {len(pcm)/sr:.1f}s — '{utt['text'][:50]}...'")

    # Human TTS generated inline in playback loop

    # Timeline
    timeline = []
    script_start = time.time()

    def log_event(**kwargs):
        kwargs["wall_time"] = round(time.time() - script_start, 3)
        timeline.append(kwargs)

    # For each agent utterance, run a streaming segment
    # Between utterances, run the speaking policy on accumulated frame logs
    total_utterances = len(episode.get("agent_utterances", []))

    for utt_idx, utt in enumerate(episode.get("agent_utterances", [])):
        utt_pcm = tts_segments[("agent", utt_idx)]
        is_last = utt_idx == total_utterances - 1

        # Check for user voice events during this utterance
        user_during = [
            uv for uv in episode.get("user_voice", [])
            if utt["time"] <= uv["start"] < utt["time"] + len(utt_pcm) / sr
        ]

        print(f"\n  [Segment {utt_idx+1}/{total_utterances}] Agent: '{utt['text'][:40]}...'")
        if user_during:
            for uv in user_during:
                print(f"    Human voice at t={uv['start']}: '{uv['text'][:40]}...'")

        # Configure streaming runtime for this segment
        save_path = None
        if args.output:
            save_path = Path(args.output).parent / f"{episode['id']}_segment{utt_idx}.jsonl"

        # Resolve room filter path (may be in corollary sibling repo)
        room_filter_path = args.room_filter
        if room_filter_path and not Path(room_filter_path).exists():
            # Try local data directory
            alt = Path("data") / Path(room_filter_path).name
            if alt.exists():
                room_filter_path = str(alt)
            else:
                print(f"    WARNING: Room filter not found at {args.room_filter}, skipping")
                room_filter_path = None

        # Resolve M4 scorer path
        m4_scorer_path = args.m4_scorer
        if m4_scorer_path and not Path(m4_scorer_path).exists():
            print(f"    WARNING: M4 scorer not found at {args.m4_scorer}")
            m4_scorer_path = None

        config = StreamingConfig(
            tts_pcm=utt_pcm,
            tts_text=utt["text"],
            input_device=args.input_device,
            output_device=args.output_device,
            sr=sr,
            callback_blocksize_ms=args.callback_blocksize_ms,
            gated_playback=True,
            m4_enable=bool(m4_scorer_path),
            m4_scorer_path=m4_scorer_path,
            m4_bank_path=args.m4_bank,
            m4_mode="auto",
            m4_self_threshold=0.50,
            room_filter_path=room_filter_path,
            energy_threshold_db=-55.0,
            similarity_threshold=0.05,
            causal_tolerance_ms=175.0,
            both_external_min_frames=3,
            both_external_energy_ratio=1.5,
            cooldown_ms=200.0,
            save_path=save_path,
        )

        runtime = StreamingRuntime(config)

        # Start user voice playback thread if scheduled during this segment
        user_threads = []
        for loop_idx, uv in enumerate(user_during):
            if args.human_output_device is not None:
                human_pcm = make_tts_pcm(uv["text"], uv["voice"])
                # Calculate delay from segment start
                delay_from_now = uv["start"] - utt["time"]
                if delay_from_now < 0:
                    delay_from_now = 0

                def _play_human(pcm=human_pcm, delay=delay_from_now):
                    time.sleep(delay)
                    import sounddevice as sd
                    sd.play(pcm, samplerate=sr, device=args.human_output_device)
                    sd.wait()

                t = threading.Thread(target=_play_human, daemon=True)
                t.start()
                user_threads.append(t)
                print(f"    Started human voice thread (delay={delay_from_now:.1f}s)")

        # Run the streaming segment
        log_event(type="segment_start", segment=utt_idx, utterance=utt["text"][:40])
        try:
            runtime.run()
        except KeyboardInterrupt:
            print("  Interrupted by user")
            break

        log_event(type="segment_end", segment=utt_idx,
                  frames_logged=runtime._frame_log_index if hasattr(runtime, '_frame_log_index') else 0)

        # After segment: run speaking policy on frame logs
        # Read frame logs and feed to tracker
        frame_log_path = save_path
        if frame_log_path and frame_log_path.exists():
            logged_actions = 0
            for line in frame_log_path.open():
                try:
                    frame = json.loads(line)
                    tracker.process_frame(frame)
                    logged_actions += 1
                except (json.JSONDecodeError, KeyError):
                    continue

            # Get current features and run policy
            features = tracker.get_features()
            action = policy.decide(features)
            log_event(type="policy_decision", segment=utt_idx, action=action,
                      context=f"after_segment_{utt_idx}")
            print(f"    Policy decision: {action} (after {logged_actions} frames)")

            # Check if this was a user-voice segment and log yield detection
            if user_during:
                # Check if controller transitioned to LISTEN
                frame_lines = list(frame_log_path.open()) if frame_log_path.exists() else []
                for line in reversed(frame_lines[-20:]):
                    try:
                        f = json.loads(line)
                        if f.get("controller_state") == "LISTEN":
                            log_event(type="yield_detected", segment=utt_idx,
                                      frame_index=f.get("frame_index"))
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

    print()
    return {
        "episode_id": episode["id"],
        "mode": "live",
        "duration_s": episode["duration_s"],
        "timeline": timeline,
        "policy": policy.__class__.__name__,
        "input_device": args.input_device,
        "output_device": args.output_device,
        "human_output_device": args.human_output_device,
    }


def evaluate_results(results: List[Dict], episodes: List[Dict]) -> Dict[str, Any]:
    """Compare live results against expected actions."""
    evaluations = []
    for result in results:
        ep_id = result["episode_id"]
        ep = next((e for e in episodes if e["id"] == ep_id), None)
        if ep is None:
            continue

        expected = ep.get("expected_actions", [])
        decisions = [e for e in result["timeline"] if e["type"] == "policy_decision"]
        actual = [e["action"] for e in decisions]

        # Action match: compare actual decisions to expected
        matches = 0
        min_len = min(len(expected), len(actual))
        for i in range(min_len):
            if actual[i] == expected[i]:
                matches += 1

        match_rate = matches / max(len(expected), 1)

        # Yield detection: check if yield happened during user voice segments
        yield_events = [e for e in result["timeline"] if e["type"] == "yield_detected"]
        expected_yields = len(ep.get("user_voice", []))

        evaluations.append({
            "episode_id": ep_id,
            "expected_actions": expected,
            "actual_actions": actual,
            "action_matches": matches,
            "action_match_rate": round(match_rate, 4),
            "yield_events": len(yield_events),
            "expected_yields": expected_yields,
            "mode": result["mode"],
            "policy": result["policy"],
        })

    overall_match = sum(e["action_match_rate"] for e in evaluations) / max(len(evaluations), 1)

    return {
        "episodes": evaluations,
        "overall_action_match_rate": round(overall_match, 4),
        "pass": overall_match >= 0.75,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 Live Hardware Validation")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None,
                        help="Output device for agent TTS (primary speaker)")
    parser.add_argument("--human-output-device", type=int, default=None,
                        help="Output device for human voice (second speaker)")
    parser.add_argument("--policy", type=str, default="heuristic",
                        choices=["heuristic", "distilled", "ppo"],
                        help="Speaking policy to evaluate")
    parser.add_argument("--policy-checkpoint", type=str, default=None,
                        help="Override checkpoint path for distilled/ppo policy")
    parser.add_argument("--episodes", type=str, default="all",
                        help="Comma-separated episode IDs or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without audio devices (validates episode scripts + policy)")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--callback-blocksize-ms", type=float, default=120.0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--m4-scorer", type=str,
                        default="/Users/juicebox/Projects/corollary/data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt",
                        help="Path to fine-tuned M4 scorer .pt file")
    parser.add_argument("--m4-bank", type=str, default=None,
                        help="Path to M4 prototype bank .npz file")
    parser.add_argument("--room-filter", type=str,
                        default="/Users/juicebox/Projects/corollary/data/m3_broca_room_filter_production.npy",
                        help="Path to room filter .npy file")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Select episodes
    if args.episodes == "all":
        episodes = LIVE_EPISODES
    else:
        ep_ids = [e.strip() for e in args.episodes.split(",")]
        episodes = [e for e in LIVE_EPISODES if e["id"] in ep_ids]

    if not episodes:
        print("ERROR: No matching episodes found")
        sys.exit(1)

    # Load policy
    print(f"Loading {args.policy} policy...")
    policy = load_speaking_policy(args.policy, args.policy_checkpoint)

    print(f"\n{'=' * 60}")
    print(f"M8 Live Hardware Validation")
    print(f"{'=' * 60}")
    print(f"  Policy:       {policy.__class__.__name__}")
    print(f"  Mode:         {'dry-run' if args.dry_run else 'live hardware'}")
    print(f"  Episodes:     {len(episodes)}")
    if not args.dry_run:
        print(f"  Input device: {args.input_device}")
        print(f"  Output device: {args.output_device}")
        if args.human_output_device:
            print(f"  Human device:  {args.human_output_device}")
    print()

    # Run episodes
    results = []
    for ep in episodes:
        print(f"\n--- Episode: {ep['id']} ---")
        print(f"  {ep['description']}")

        if args.dry_run:
            result = run_episode_dry(ep, policy)
        else:
            if args.input_device is None or args.output_device is None:
                print("ERROR: --input-device and --output-device required for live mode")
                print("  Use --dry-run to test without hardware")
                print("  Use --list-devices to see available devices")
                sys.exit(1)
            result = run_episode_live(ep, policy, args)

        results.append(result)

        # Summary
        decisions = [e for e in result["timeline"] if e["type"] == "policy_decision"]
        print(f"  Decisions: {[d['action'] for d in decisions]}")
        print(f"  Expected:  {ep.get('expected_actions', [])}")
        expected = ep.get("expected_actions", [])
        actual = [d["action"] for d in decisions]
        matches = sum(1 for i in range(min(len(expected), len(actual))) if actual[i] == expected[i])
        print(f"  Match:     {matches}/{len(expected)} ({matches/max(len(expected),1):.0%})")

    # Evaluate
    print(f"\n{'=' * 60}")
    print(f"Evaluation")
    print(f"{'=' * 60}")

    evals = evaluate_results(results, episodes)
    for e in evals["episodes"]:
        status = "PASS" if e["action_match_rate"] >= 0.75 else "INFO"
        print(f"  [{status}] {e['episode_id']:30s}  match={e['action_match_rate']:.0%}  "
              f"actions={e['actual_actions']}")

    print(f"\n  Overall match rate: {evals['overall_action_match_rate']:.0%}")

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results, "evaluation": evals}, f, indent=2)
        print(f"\n  Saved to {args.output}")

    sys.exit(0 if evals["pass"] else 1)


if __name__ == "__main__":
    main()
