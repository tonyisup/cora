#!/usr/bin/env python3
"""M2 — Data Collection for Distillation.

Collects training data for the distilled speaking policy by:
1. Running the deterministic simulator and saving every (perception, action) pair
2. Querying a teacher LLM (Claude) on a subset of frames for soft labels
3. Generating hard-negative edge cases for timing ambiguity

Output: JSONL files compatible with scripts/m2_train.py

Usage:
  .venv/bin/python scripts/m2_collect_data.py --mode deterministic --trials 150 --output results/m2_data_deterministic.jsonl
  .venv/bin/python scripts/m2_collect_data.py --mode teacher --limit 5000 --output results/m2_data_teacher.jsonl
  .venv/bin/python scripts/m2_collect_data.py --mode hard_negatives --count 500 --output results/m2_data_hardneg.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent))

from m1_speaking_policy_sim import (
    Action,
    AgentState,
    PerceptionFrame,
    SimulatorConfig,
    SpeakingPolicySimulator,
)


# ── Feature Extraction ──────────────────────────────────────────────────────

STATE_INDEX = {s.value: i for i, s in enumerate(AgentState)}
ATTR_INDEX = {"silence": 0, "self": 1, "world": 1, "both": 2}
ACTION_INDEX = {a.value: i for i, a in enumerate(Action)}


def frame_to_features(frame: PerceptionFrame, policy_state: str, has_queued: bool, user_speaking: bool, agent_speaking: bool) -> Dict[str, Any]:
    """Convert a PerceptionFrame + policy state into a flat feature dict."""
    return {
        "time_ms": frame.time_ms,
        "state": policy_state,
        "state_idx": STATE_INDEX.get(policy_state, 0),
        "attribution": frame.attribution,
        "attr_idx": ATTR_INDEX.get(frame.attribution, 0),
        "silence_duration_ms": frame.silence_duration_ms,
        "silence_duration_log": _log_scale(frame.silence_duration_ms),
        "vad_energy_db": frame.vad_energy_db,
        "has_queued_utterance": has_queued,
        "user_speaking": user_speaking,
        "agent_speaking": agent_speaking,
        "asr_text": frame.asr_text,
        "speaker": frame.speaker,
    }


def _log_scale(v: float) -> float:
    """Log-scale a non-negative scalar for neural net friendliness."""
    import math
    return math.log1p(v / 1000.0)  # convert ms to seconds, then log1p


# ── Mode 1: Deterministic Data Collection ───────────────────────────────────

def collect_deterministic(trials: int, seeds: List[int], output_path: Path, verbose: bool = False) -> int:
    """Run deterministic simulator and save every (features, action) pair."""
    written = 0

    with output_path.open("w") as f:
        for trial_id in range(trials):
            seed = seeds[trial_id] if trial_id < len(seeds) else seeds[-1] + trial_id
            config = SimulatorConfig(
                num_trials=1,
                seed=seed,
                policy_type="deterministic",
                max_turns_per_trial=200,
                max_trial_duration_ms=60000.0,
                verbose=False,
            )
            sim = SpeakingPolicySimulator(config)
            count = _run_simulation_collect(sim, f, source=f"det_trial_{trial_id}")
            written += count

            if verbose and (trial_id + 1) % 10 == 0:
                print(f"  Collected {written} frames from {trial_id + 1}/{trials} trials...")

    return written


def _run_simulation_collect(sim: SpeakingPolicySimulator, output_file, source: str) -> int:
    """Run a single simulation and write all (features, action) pairs to output_file."""
    # We replicate the trial loop from SpeakingPolicySimulator.run_trial
    # but hook into every policy decision to log features + action.
    import io

    sim.policy.reset()
    sim.perception.reset()
    sim.user.reset(seed=sim.config.seed)

    time_ms = 0.0
    turn_count = 0
    agent_speaking = False
    user_speaking = False
    agent_speech_text = ""
    agent_speech_start_ms = 0.0
    written = 0

    if not sim.user.done:
        sim.policy.queue_utterance(sim.user.agent_utterance)

    while time_ms < sim.config.max_trial_duration_ms and turn_count < sim.config.max_turns_per_trial:
        # 1. User tick
        user_event = sim.user.tick(time_ms, sim.config.time_step_ms)
        user_info = sim.user.get_perception(time_ms)

        if user_event:
            if user_event["type"] in ("user_speech_start", "user_interrupt"):
                user_speaking = True
                sim.policy.on_user_text(user_event["text"])
                if user_event["type"] == "user_interrupt" and agent_speaking:
                    sim.policy.on_yield(time_ms)
                    agent_speaking = False
                turn_count += 1
            elif user_event["type"] == "user_speech_end":
                user_speaking = False
                if not sim.user.done:
                    sim.policy.queue_utterance(sim.user.agent_utterance)

        # 2. Perception frame
        asr_text = ""
        speaker = ""
        if user_speaking:
            asr_text = sim.user._current_response
            speaker = "user"
        elif agent_speaking:
            speaker = "agent"

        frame = sim.perception.frame(
            time_ms=time_ms,
            agent_speaking=agent_speaking,
            user_speaking=user_speaking,
            asr_text=asr_text,
            speaker=speaker,
        )

        # 3. Policy decision
        agent_script_line = sim.user.agent_utterance if not sim.user.done else ""
        action, text = sim.policy.decide(frame, user_info, agent_script_line)

        # 4. Log this decision
        policy_state = sim.policy.state.value if hasattr(sim.policy, 'state') else "unknown"
        has_queued = bool(sim.policy._queued_utterance) if hasattr(sim.policy, '_queued_utterance') else False

        features = frame_to_features(frame, policy_state, has_queued, user_speaking, agent_speaking)
        record = {
            "source": source,
            "features": features,
            "action": action.value,
            "action_idx": ACTION_INDEX.get(action.value, -1),
            "text": text,
        }
        output_file.write(json.dumps(record, default=str) + "\n")
        written += 1

        # 5. Execute action
        if action == Action.SPEAK and text and not agent_speaking:
            agent_speaking = True
            agent_speech_text = text
            agent_speech_start_ms = time_ms
            sim.policy.on_speech_start(text, time_ms)
            sim.user.on_agent_speech_start(text, time_ms)
        elif action == Action.YIELD and agent_speaking:
            agent_speaking = False
            sim.policy.on_yield(time_ms)
        elif action == Action.REPAIR:
            sim.policy.on_speech_start(text, time_ms)
            time_ms += 1000.0
            sim.policy.on_speech_end(text, time_ms)
            sim.user.on_agent_speech_end(text, 1000.0, time_ms)
            agent_speaking = False
        elif action == Action.CLARIFY:
            sim.policy.on_speech_start(text, time_ms)
            time_ms += 1200.0
            sim.policy.on_speech_end(text, time_ms)
            sim.user.on_agent_speech_end(text, 1200.0, time_ms)
            agent_speaking = False

        time_ms += sim.config.time_step_ms

    return written


# ── Mode 2: Teacher Query ───────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEACHER_SYSTEM_PROMPT = """You are a speaking policy for a conversational AI agent. Your job is to decide when the agent should speak, wait, clarify, repair, or yield in a turn-taking conversation.

Rules:
- Speak only when it's the agent's turn (user has finished speaking + brief silence + queued utterance exists)
- Wait while the user is speaking, or when there is silence but nothing queued
- Yield immediately if the user interrupts the agent mid-sentence (attribution=both while agent is speaking)
- Repair (say "Sorry, go ahead.") if already in the repair state
- Clarify (ask user to repeat) if the agent asked a question and the user didn't respond after a long silence

Few-shot examples:

Example 1 — speak_on_silence:
  State: listen, attribution: silence, silence: 500ms, queued: yes
  → ACTION: speak | TEXT: <queued utterance>

Example 2 — yield_on_collision:
  State: speak, attribution: both, silence: 0ms, queued: no
  → ACTION: yield | TEXT: none

Example 3 — clarify_after_silence:
  State: listen, attribution: silence, silence: 2500ms, queued: no, last agent: "Could you repeat that?"
  → ACTION: clarify | TEXT: Sorry — could you repeat that?

Example 4 — wait_while_user_speaks:
  State: listen, attribution: world, silence: 120ms, queued: yes
  → ACTION: wait | TEXT: none

Example 5 — repair_from_repair_state:
  State: repair, attribution: silence, queued: no, repair_queued: "Sorry, go ahead."
  → ACTION: repair | TEXT: Sorry, go ahead.

Reply exactly with:
ACTION: <speak|wait|clarify|repair|yield>
EXPLANATION: <1 sentence>
TEXT: <the exact text to speak, or "none">"""


def read_openrouter_key() -> str:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        raise RuntimeError(f"missing OpenRouter env file: {env_path}")
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in ~/.hermes/.env")


def query_teacher(features: Dict[str, Any], model: str = "anthropic/claude-3.5-haiku", timeout_s: float = 30.0) -> Optional[Dict[str, Any]]:
    """Query the teacher LLM with a feature frame. Returns {action, text, explanation} or None."""
    key = read_openrouter_key()

    user_prompt = f"""Current conversation state:
- Agent state: {features.get('state', 'unknown')}
- Attribution: {features.get('attribution', 'unknown')}
- Silence duration: {features.get('silence_duration_ms', 0):.0f}ms
- VAD energy: {features.get('vad_energy_db', -60):.1f} dB
- Has queued utterance: {features.get('has_queued_utterance', False)}
- User speaking: {features.get('user_speaking', False)}
- Agent speaking: {features.get('agent_speaking', False)}

What should the agent do next? Respond with:
ACTION: <speak|wait|clarify|repair|yield>
EXPLANATION: <1 sentence>
TEXT: <text or "none">"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 120,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or msg.get("reasoning") or ""
            parsed = _parse_teacher_response(content.strip())
            if parsed:
                parsed["raw_response"] = content.strip()
                parsed["model_used"] = data.get("model", model)
                return parsed
        return None
    except Exception as e:
        print(f"  WARNING: teacher query failed: {e}")
        return None


def _parse_teacher_response(text: str) -> Optional[Dict[str, str]]:
    """Parse ACTION/EXPLANATION/TEXT from teacher response."""
    action = None
    explanation = ""
    response_text = "none"
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TEXT:"):
            response_text = line.split(":", 1)[1].strip()
            if response_text.lower() in ('none', '"none"', ''):
                response_text = "none"
    if action in ("speak", "wait", "clarify", "repair", "yield"):
        return {"action": action, "explanation": explanation, "text": response_text}
    return None


def collect_teacher(input_path: Path, limit: int, output_path: Path, model: str = "anthropic/claude-3.5-haiku", sample_rate: float = 1.0):
    """Read deterministic data, sample frames, query teacher, write results."""
    # Read all features from deterministic data
    all_records = []
    with input_path.open() as f:
        for line in f:
            all_records.append(json.loads(line))

    if sample_rate < 1.0:
        random.shuffle(all_records)
        all_records = all_records[:int(len(all_records) * sample_rate)]

    # Sample uniformly across the dataset
    if limit and limit < len(all_records):
        indices = sorted(random.sample(range(len(all_records)), limit))
        sampled = [all_records[i] for i in indices]
    else:
        sampled = all_records

    print(f"Querying teacher ({model}) on {len(sampled)} frames...")

    written = 0
    errors = 0
    with output_path.open("w") as f:
        for i, record in enumerate(sampled):
            features = record["features"]
            teacher_output = query_teacher(features, model=model)

            if teacher_output:
                out_record = {
                    "source": f"teacher_{model}",
                    "features": features,
                    "action": teacher_output["action"],
                    "action_idx": ACTION_INDEX.get(teacher_output["action"], -1),
                    "text": teacher_output.get("text", "none"),
                    "explanation": teacher_output.get("explanation", ""),
                    "teacher_model": teacher_output.get("model_used", model),
                    "deterministic_action": record["action"],
                    "agreement": teacher_output["action"] == record["action"],
                }
                f.write(json.dumps(out_record, default=str) + "\n")
                written += 1
            else:
                errors += 1

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(sampled)} queries, {written} successful, {errors} errors")

            # Rate limiting: be nice to the API
            time.sleep(0.1)

    print(f"Done: {written} teacher labels written, {errors} errors")


# ── Mode 3: Hard-Negative Generation ────────────────────────────────────────

HARD_NEGATIVE_TEMPLATES = [
    {
        "name": "short_pause_mid_turn",
        "description": "User pauses briefly mid-sentence (300ms). Agent should wait.",
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 300.0,
        "vad_energy_db": -55.0,
        "has_queued_utterance": True,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "wait",
    },
    {
        "name": "post_response_silence",
        "description": "User just answered and went quiet. Agent should wait (not clarify).",
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 1500.0,
        "vad_energy_db": -52.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "wait",
    },
    {
        "name": "rapid_turn_taking",
        "description": "Back-and-forth with very short gaps. Agent should speak.",
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 150.0,
        "vad_energy_db": -50.0,
        "has_queued_utterance": True,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "speak",
    },
    {
        "name": "collision_immediate_yield",
        "description": "Both speaking simultaneously. Agent should yield.",
        "state": "speak",
        "attribution": "both",
        "silence_duration_ms": 0.0,
        "vad_energy_db": -20.0,
        "has_queued_utterance": False,
        "user_speaking": True,
        "agent_speaking": True,
        "expected_action": "yield",
    },
    {
        "name": "interrupt_mid_sentence_yield",
        "description": "User interrupts while the agent is speaking. Agent must yield immediately.",
        "state": "speak",
        "attribution": "world",
        "silence_duration_ms": 0.0,
        "vad_energy_db": -18.0,
        "has_queued_utterance": False,
        "user_speaking": True,
        "agent_speaking": True,
        "expected_action": "yield",
    },
    {
        "name": "repair_state_forced",
        "description": "In repair state with repair utterance queued. Must repair.",
        "state": "repair",
        "attribution": "silence",
        "silence_duration_ms": 0.0,
        "vad_energy_db": -60.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "repair",
    },
    {
        "name": "clarify_after_no_response",
        "description": "Agent asked question, user didn't respond after 2.5s. Must clarify.",
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 2500.0,
        "vad_energy_db": -55.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "clarify",
    },
    {
        "name": "wait_while_user_active",
        "description": "User is actively speaking. Agent must wait even with queued utterance.",
        "state": "listen",
        "attribution": "world",
        "silence_duration_ms": 50.0,
        "vad_energy_db": -18.0,
        "has_queued_utterance": True,
        "user_speaking": True,
        "agent_speaking": False,
        "expected_action": "wait",
    },
    {
        "name": "idle_no_user_yet",
        "description": "Idle state, no user speech yet, no queued utterance. Wait.",
        "state": "idle",
        "attribution": "silence",
        "silence_duration_ms": 5000.0,
        "vad_energy_db": -60.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "wait",
    },
    {
        "name": "idle_no_queue_silence",
        "description": "Idle/listen state, no queue, short silence, no user speech. Agent should wait rather than clarify.",
        "state": "listen",
        "attribution": "silence",
        "silence_duration_ms": 150.0,
        "vad_energy_db": -58.0,
        "has_queued_utterance": False,
        "user_speaking": False,
        "agent_speaking": False,
        "expected_action": "wait",
    },
]


def collect_hard_negatives(count: int, output_path: Path):
    """Generate hard-negative edge cases with randomized timing noise."""
    records = []
    for i in range(count):
        template = random.choice(HARD_NEGATIVE_TEMPLATES)
        # Add timing noise
        noise_ms = random.gauss(0, 100)
        silence_ms = max(0, template["silence_duration_ms"] + noise_ms)
        vad_noise = random.gauss(0, 2)
        vad_db = template["vad_energy_db"] + vad_noise

        features = {
            "time_ms": 0.0,
            "state": template["state"],
            "state_idx": STATE_INDEX.get(template["state"], 0),
            "attribution": template["attribution"],
            "attr_idx": ATTR_INDEX.get(template["attribution"], 0),
            "silence_duration_ms": silence_ms,
            "silence_duration_log": _log_scale(silence_ms),
            "vad_energy_db": vad_db,
            "has_queued_utterance": template["has_queued_utterance"],
            "user_speaking": template["user_speaking"],
            "agent_speaking": template["agent_speaking"],
            "asr_text": "",
            "speaker": "",
        }
        record = {
            "source": f"hardneg_{template['name']}",
            "features": features,
            "action": template["expected_action"],
            "action_idx": ACTION_INDEX.get(template["expected_action"], -1),
            "text": "",
            "description": template["description"],
        }
        records.append(record)

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")

    print(f"Generated {count} hard-negative examples ({len(set(r['source'] for r in records))} unique templates)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="M2 data collection for distillation")
    parser.add_argument("--mode", required=True, choices=["deterministic", "teacher", "hard_negatives", "all"])
    parser.add_argument("--trials", type=int, default=150, help="Number of deterministic trials")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds (default: 0..trials-1)")
    parser.add_argument("--limit", type=int, default=5000, help="Max teacher queries")
    parser.add_argument("--count", type=int, default=500, help="Hard-negative count")
    parser.add_argument("--output", type=str, default=None, help="Output path (or prefix for --mode all)")
    parser.add_argument("--input", type=str, default=None, help="Input path (for --mode teacher, default: results/m2_data_deterministic.jsonl)")
    parser.add_argument("--teacher-model", type=str, default="anthropic/claude-3.5-haiku")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "all":
        # Run all three modes with a shared prefix
        prefix = args.output or f"results/m2_data_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"
        base = Path(prefix)
        det_path = base.parent / f"{base.stem}_deterministic.jsonl"
        teach_path = base.parent / f"{base.stem}_teacher.jsonl"
        hn_path = base.parent / f"{base.stem}_hardneg.jsonl"

        print("=== Mode 1: Deterministic ===")
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.trials))
        n = collect_deterministic(args.trials, seeds, det_path, verbose=args.verbose)
        print(f"Collected {n} deterministic frames → {det_path}")

        print("\n=== Mode 2: Teacher ===")
        collect_teacher(det_path, args.limit, teach_path, model=args.teacher_model)

        print("\n=== Mode 3: Hard Negatives ===")
        collect_hard_negatives(args.count, hn_path)

        print(f"\nAll data collected. Merge with: cat {det_path} {teach_path} {hn_path} > {base}_merged.jsonl")
        return 0

    if args.mode == "deterministic":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.trials))
        out = Path(args.output) if args.output else Path("results/m2_data_deterministic.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        n = collect_deterministic(args.trials, seeds, out, verbose=args.verbose)
        print(f"Collected {n} frames → {out}")
        return 0

    if args.mode == "teacher":
        input_path = Path(args.input) if args.input else Path("results/m2_data_deterministic.jsonl")
        out = Path(args.output) if args.output else Path("results/m2_data_teacher.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        collect_teacher(input_path, args.limit, out, model=args.teacher_model)
        return 0

    if args.mode == "hard_negatives":
        out = Path(args.output) if args.output else Path("results/m2_data_hardneg.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        collect_hard_negatives(args.count, out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
