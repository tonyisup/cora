#!/usr/bin/env python3
"""M3 — Local-LLM user-driven data collection.

This script reuses the M1 speaking-policy simulator primitives but replaces the
scripted user with a local LLM-backed user simulator. The result is a JSONL
stream of (features, action) records compatible with the M2 training stack.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

sys.path.insert(0, str(Path(__file__).parent))

from m1_speaking_policy_sim import (
    Action,
    AgentState,
    PerceptionFrame,
    SimulatorConfig,
    SpeakingPolicySimulator,
    CONVERSATION_SCRIPTS,
)
from m2_collect_data import ACTION_INDEX, ATTR_INDEX, STATE_INDEX, _log_scale, frame_to_features
from m3_user_agent import M3UserAgent, UserAgentPacket


class LocalLLMUser:
    """A simulator-facing user that asks a local LLM how to respond on each turn."""

    def __init__(self, model: str, timeout_s: float = 45.0):
        self.agent = M3UserAgent(model=model, timeout_s=timeout_s)
        self.rng = random.Random(0)
        self.state = "silent"
        self._script: List[tuple[str, str]] = []
        self._turn_idx = 0
        self._current_response = ""
        self._response_delay_ms = 0.0
        self._turn_duration_ms = 0.0
        self._speaking_start_ms = 0.0
        self._will_interrupt = False
        self._will_overlap = False
        self._will_not_respond = False
        self._interrupt_at_ms = 0.0
        self._agent_speaking = False
        self._history: List[Dict[str, str]] = []
        self._last_agent_utterance = ""
        self._last_user_utterance = ""
        self._seed = 0
        self._scenario_id = 0
        self._scenario_name = "m3_local_user"

    def reset(self, script_id: Optional[int] = None, seed: int = 0):
        self.rng = random.Random(seed)
        self.state = "silent"
        self._turn_idx = 0
        self._current_response = ""
        self._response_delay_ms = 0.0
        self._turn_duration_ms = 0.0
        self._speaking_start_ms = 0.0
        self._will_interrupt = False
        self._will_overlap = False
        self._will_not_respond = False
        self._interrupt_at_ms = 0.0
        self._agent_speaking = False
        self._history = []
        self._last_agent_utterance = ""
        self._last_user_utterance = ""
        self._seed = seed
        self._scenario_id = script_id or 0
        self._scenario_name = f"m3_script_{self._scenario_id}"
        self._script = CONVERSATION_SCRIPTS[self._scenario_id % len(CONVERSATION_SCRIPTS)] if script_id is not None else self.rng.choice(CONVERSATION_SCRIPTS)

    @property
    def agent_utterance(self) -> str:
        if self._turn_idx < len(self._script):
            return self._script[self._turn_idx][0]
        return ""

    @property
    def done(self) -> bool:
        return self._turn_idx >= len(self._script)

    def _packet(self, turn_index: int) -> UserAgentPacket:
        return UserAgentPacket(
            scenario_id=self._scenario_id,
            scenario_name=self._scenario_name,
            turn_index=turn_index,
            history=self._history[-8:],
            last_assistant_utterance=self._last_agent_utterance,
            last_user_utterance=self._last_user_utterance,
            goal=f"Conversation {self._scenario_name}",
            style="natural",
            allow_silence=True,
            allow_interrupt=True,
            allow_overlap=True,
            seed=self._seed * 1000 + turn_index,
        )

    def _decide_next(self, turn_index: int):
        result = self.agent.generate(self._packet(turn_index))
        mode = result.mode
        self._current_response = result.text if mode != "silence" else ""
        self._will_interrupt = mode == "interrupt"
        self._will_overlap = mode == "overlap"
        self._will_not_respond = mode == "silence"
        self._response_delay_ms = 0.0
        self._turn_duration_ms = 0.0
        if mode in ("reply", "interrupt", "overlap"):
            # Keep these short to keep traces moving, but variable enough to be interesting.
            self._response_delay_ms = max(50.0, min(1500.0, result.latency_s * 250.0 + self.rng.uniform(120.0, 700.0)))
            self._turn_duration_ms = max(400.0, min(2500.0, 450.0 + len(self._current_response) * 18.0))
        elif mode == "silence":
            self._response_delay_ms = max(800.0, self.rng.uniform(900.0, 3200.0))
            self._turn_duration_ms = 0.0

    def on_agent_speech_start(self, text: str, time_ms: float):
        self._agent_speaking = True
        self._last_agent_utterance = text
        self._history.append({"speaker": "assistant", "text": text})
        self._decide_next(self._turn_idx)
        if self._will_interrupt:
            self._interrupt_at_ms = time_ms + max(200.0, self._response_delay_ms)
        if self._will_overlap:
            self.state = "speaking"

    def on_agent_speech_end(self, text: str, duration_ms: float, time_ms: float):
        self._agent_speaking = False
        if self._will_not_respond:
            return 0.0
        if not self._will_interrupt and not self._will_overlap:
            self.state = "speaking"
        return self._response_delay_ms

    def on_agent_yield(self, time_ms: float):
        self._agent_speaking = False
        self._response_delay_ms = 50.0
        self._turn_duration_ms = max(self._turn_duration_ms, 400.0)

    def get_perception(self, time_ms: float) -> Dict[str, Any]:
        return {
            "user_speaking": self.state == "speaking",
            "user_interrupting": self._will_interrupt and self.state == "speaking",
            "user_text": self._current_response if self.state == "speaking" else "",
        }

    def tick(self, time_ms: float, dt_ms: float) -> Optional[Dict[str, Any]]:
        if self.done:
            return None

        if self._will_overlap and self.state == "silent" and self._agent_speaking:
            self._will_overlap = False
            self.state = "speaking"
            self._speaking_start_ms = time_ms
            return {
                "type": "user_speech_start",
                "text": self._current_response,
                "time_ms": time_ms,
                "will_clarify": False,
                "will_correct": False,
            }

        if self._will_interrupt and self.state == "silent" and time_ms >= self._interrupt_at_ms:
            self._will_interrupt = False
            self.state = "speaking"
            self._speaking_start_ms = time_ms
            return {
                "type": "user_interrupt",
                "text": self._current_response,
                "time_ms": time_ms,
                "will_clarify": False,
                "will_correct": False,
            }

        if self.state == "silent":
            if self._response_delay_ms > 0:
                self._response_delay_ms -= dt_ms
                if self._response_delay_ms <= 0:
                    if self._will_not_respond:
                        return None
                    self.state = "speaking"
                    self._speaking_start_ms = time_ms
                    self._last_user_utterance = self._current_response
                    self._history.append({"speaker": "user", "text": self._current_response})
                    return {
                        "type": "user_speech_start",
                        "text": self._current_response,
                        "time_ms": time_ms,
                        "will_clarify": False,
                        "will_correct": False,
                    }

        elif self.state == "speaking":
            elapsed = time_ms - self._speaking_start_ms
            if elapsed >= self._turn_duration_ms:
                self.state = "silent"
                self._turn_idx += 1
                self._last_user_utterance = self._current_response
                return {
                    "type": "user_speech_end",
                    "text": self._current_response,
                    "time_ms": time_ms,
                    "duration_ms": elapsed,
                }

        return None


def run_collect(trials: int, seed: int, output_path: Path, model: str, timeout_s: float) -> int:
    written = 0
    with output_path.open("w") as f:
        for trial_id in range(trials):
            config = SimulatorConfig(
                num_trials=1,
                seed=seed + trial_id,
                policy_type="deterministic",
                max_turns_per_trial=40,
                max_trial_duration_ms=30000.0,
                verbose=False,
            )
            sim = SpeakingPolicySimulator(config)
            sim.user = cast(Any, LocalLLMUser(model=model, timeout_s=timeout_s))
            sim.user.reset(script_id=trial_id % len(CONVERSATION_SCRIPTS), seed=seed + trial_id)
            written += _run_single(sim, f, source=f"m3_trial_{trial_id}")
    return written


def _run_single(sim: SpeakingPolicySimulator, output_file, source: str) -> int:
    sim.policy.reset()
    sim.perception.reset()

    time_ms = 0.0
    turn_count = 0
    agent_speaking = False
    user_speaking = False
    written = 0

    if not sim.user.done:
        sim.policy.queue_utterance(sim.user.agent_utterance)

    while time_ms < sim.config.max_trial_duration_ms and turn_count < sim.config.max_turns_per_trial:
        user_event = sim.user.tick(time_ms, sim.config.time_step_ms)
        user_info = sim.user.get_perception(time_ms)

        if user_event:
            if user_event["type"] in ("user_speech_start", "user_interrupt"):
                user_speaking = True
                sim.policy.on_user_text(user_event["text"])
                if user_event["type"] == "user_interrupt" and agent_speaking:
                    sim.policy.on_yield(time_ms)
                    sim.user.on_agent_yield(time_ms)
                    agent_speaking = False
                turn_count += 1
            elif user_event["type"] == "user_speech_end":
                user_speaking = False
                if not sim.user.done:
                    sim.policy.queue_utterance(sim.user.agent_utterance)

        asr_text = sim.user._current_response if user_speaking else ("" if not agent_speaking else "")
        speaker = "user" if user_speaking else ("agent" if agent_speaking else "")
        frame = sim.perception.frame(
            time_ms=time_ms,
            agent_speaking=agent_speaking,
            user_speaking=user_speaking,
            asr_text=asr_text,
            speaker=speaker,
        )

        agent_script_line = sim.user.agent_utterance if not sim.user.done else ""
        action, text = sim.policy.decide(frame, user_info, agent_script_line)

        policy_state = sim.policy.state.value if hasattr(sim.policy, "state") else "unknown"
        has_queued = bool(sim.policy._queued_utterance) if hasattr(sim.policy, "_queued_utterance") else False
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

        if action == Action.SPEAK and text and not agent_speaking:
            agent_speaking = True
            sim.policy.on_speech_start(text, time_ms)
            sim.user.on_agent_speech_start(text, time_ms)
        elif action == Action.YIELD and agent_speaking:
            agent_speaking = False
            sim.policy.on_yield(time_ms)
            sim.user.on_agent_yield(time_ms)
        elif action == Action.REPAIR:
            sim.policy.on_speech_start(text, time_ms)
            time_ms += 1000.0
            sim.policy.on_speech_end(text, time_ms)
            agent_speaking = False
        elif action == Action.CLARIFY:
            sim.policy.on_speech_start(text, time_ms)
            time_ms += 1200.0
            sim.policy.on_speech_end(text, time_ms)
            agent_speaking = False

        time_ms += sim.config.time_step_ms

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect M3 traces with a local LLM user")
    parser.add_argument("--output", default="results/m3_data_local_user.jsonl", help="Output JSONL path")
    parser.add_argument("--trials", type=int, default=12, help="Number of simulated conversations")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--model", default="gemma-4-e4b-it-4bit", help="Local OMLX model")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-turn timeout in seconds")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = run_collect(args.trials, args.seed, out_path, args.model, args.timeout)
    print(f"Collected {written} frames -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
