#!/usr/bin/env python3
"""M1 — Speaking Policy Simulator (Phase 1: Harness + Scripted User).

Offline simulated environment for evaluating speaking policies.
No live audio hardware. All TTS is synthetic via Broca (edge-tts).

The simulator runs N trials of a conversation between:
  - A speaking policy (deterministic state machine or prompt-based)
  - A scripted user agent (Tier 1: deterministic FSM)

Each trial produces per-turn metrics. Aggregated metrics are written as JSON.

Usage:
    .venv/bin/python scripts/m1_speaking_policy_sim.py
    .venv/bin/python scripts/m1_speaking_policy_sim.py --trials 50
    .venv/bin/python scripts/m1_speaking_policy_sim.py --output /tmp/m1_results.json
    .venv/bin/python scripts/m1_speaking_policy_sim.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Package path setup ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

# broca.tts is available via sys.path; imported lazily to avoid
# edge-tts dependency during harness testing.
# from broca.tts import synthesize as broca_synthesize  # noqa: F401

# ── Constants ───────────────────────────────────────────────────────────────
SR = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SR * VAD_FRAME_MS / 1000)

# Agent action set
class Action(str, Enum):
    SPEAK = "speak"
    WAIT = "wait"
    CLARIFY = "clarify"
    REPAIR = "repair"
    YIELD = "yield"

# Agent state
class AgentState(str, Enum):
    IDLE = "idle"
    LISTEN = "listen"
    SPEAK = "speak"
    REPAIR = "repair"

# User state
class UserState(str, Enum):
    SILENT = "silent"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """Record of a single conversational turn."""
    turn_idx: int
    speaker: str           # "agent" or "user"
    action: Optional[str]  # Action enum value or user behavior label
    text: str
    start_ms: float
    end_ms: float
    was_interruption: bool = False
    was_collision: bool = False
    was_repair: bool = False
    was_clarification: bool = False
    gap_before_ms: float = 0.0  # silence before this turn started

@dataclass
class TrialMetrics:
    """Metrics for a single trial (one conversation)."""
    trial_id: int
    total_turns: int = 0
    agent_turns: int = 0
    user_turns: int = 0
    interruptions: int = 0
    collisions: int = 0
    repairs_attempted: int = 0
    repairs_successful: int = 0
    clarifications: int = 0
    clarifications_successful: int = 0
    smooth_turns: int = 0
    response_gaps_ms: List[float] = field(default_factory=list)
    turn_records: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    completed: bool = False

    @property
    def interruption_rate(self) -> float:
        return self.interruptions / max(self.agent_turns, 1)

    @property
    def turn_continuation_rate(self) -> float:
        return self.smooth_turns / max(self.total_turns - 1, 1)

    @property
    def repair_success_rate(self) -> float:
        return self.repairs_successful / max(self.repairs_attempted, 1)

    @property
    def clarification_efficiency(self) -> float:
        return self.clarifications_successful / max(self.clarifications, 1)

    @property
    def collision_rate(self) -> float:
        return self.collisions / max(self.agent_turns, 1)

    @property
    def avg_response_gap_ms(self) -> float:
        return sum(self.response_gaps_ms) / max(len(self.response_gaps_ms), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "total_turns": self.total_turns,
            "agent_turns": self.agent_turns,
            "user_turns": self.user_turns,
            "interruptions": self.interruptions,
            "collisions": self.collisions,
            "repairs_attempted": self.repairs_attempted,
            "repairs_successful": self.repairs_successful,
            "clarifications": self.clarifications,
            "clarifications_successful": self.clarifications_successful,
            "smooth_turns": self.smooth_turns,
            "interruption_rate": round(self.interruption_rate, 4),
            "turn_continuation_rate": round(self.turn_continuation_rate, 4),
            "repair_success_rate": round(self.repair_success_rate, 4),
            "clarification_efficiency": round(self.clarification_efficiency, 4),
            "collision_rate": round(self.collision_rate, 4),
            "avg_response_gap_ms": round(self.avg_response_gap_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "completed": self.completed,
            "turn_records": self.turn_records,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across N trials."""
    num_trials: int = 0
    completed_trials: int = 0
    avg_interruption_rate: float = 0.0
    avg_turn_continuation_rate: float = 0.0
    avg_repair_success_rate: float = 0.0
    avg_clarification_efficiency: float = 0.0
    avg_collision_rate: float = 0.0
    avg_response_gap_ms: float = 0.0
    avg_duration_ms: float = 0.0
    interruption_rate_std: float = 0.0
    turn_continuation_rate_std: float = 0.0
    per_trial: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_trials": self.num_trials,
            "completed_trials": self.completed_trials,
            "avg_interruption_rate": round(self.avg_interruption_rate, 4),
            "avg_turn_continuation_rate": round(self.avg_turn_continuation_rate, 4),
            "avg_repair_success_rate": round(self.avg_repair_success_rate, 4),
            "avg_clarification_efficiency": round(self.avg_clarification_efficiency, 4),
            "avg_collision_rate": round(self.avg_collision_rate, 4),
            "avg_response_gap_ms": round(self.avg_response_gap_ms, 1),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "interruption_rate_std": round(self.interruption_rate_std, 4),
            "turn_continuation_rate_std": round(self.turn_continuation_rate_std, 4),
            "per_trial": self.per_trial,
        }


# ── Simulated Perception ────────────────────────────────────────────────────
# In the offline simulator, we know ground truth (who is speaking when).
# The perception layer simulates what the real pipeline would report.

@dataclass
class PerceptionFrame:
    """Simulated perception output for one time step."""
    time_ms: float
    agent_speaking: bool      # ground truth: is agent emitting audio?
    user_speaking: bool        # ground truth: is user emitting audio?
    vad_energy_db: float       # simulated VAD energy
    silence_duration_ms: float # how long since last speech ended
    attribution: str           # "silence", "self", "world", "both"
    asr_text: str              # transcribed text (empty if no speech)
    speaker: str               # "agent", "user", or ""


class SimulatedPerception:
    """Generates perception frames from ground-truth speaking state."""

    def __init__(self, vad_noise_db: float = 3.0):
        self.vad_noise_db = vad_noise_db
        self._silence_start_ms: Optional[float] = None

    def frame(
        self,
        time_ms: float,
        agent_speaking: bool,
        user_speaking: bool,
        asr_text: str = "",
        speaker: str = "",
    ) -> PerceptionFrame:
        # Determine attribution
        if agent_speaking and user_speaking:
            attribution = "both"
        elif agent_speaking:
            attribution = "self"
        elif user_speaking:
            attribution = "world"
        else:
            attribution = "silence"

        # Simulate VAD energy
        if agent_speaking or user_speaking:
            base_energy = random.uniform(-25.0, -15.0)
        else:
            base_energy = random.uniform(-60.0, -50.0)
        vad_energy_db = base_energy + random.gauss(0, self.vad_noise_db)

        # Track silence duration
        if not agent_speaking and not user_speaking:
            if self._silence_start_ms is None:
                self._silence_start_ms = time_ms
            silence_duration_ms = time_ms - self._silence_start_ms
        else:
            self._silence_start_ms = None
            silence_duration_ms = 0.0

        return PerceptionFrame(
            time_ms=time_ms,
            agent_speaking=agent_speaking,
            user_speaking=user_speaking,
            vad_energy_db=vad_energy_db,
            silence_duration_ms=silence_duration_ms,
            attribution=attribution,
            asr_text=asr_text,
            speaker=speaker,
        )

    def reset(self):
        self._silence_start_ms = None


# ── Scripted User Agent (Tier 1) ────────────────────────────────────────────

@dataclass
class UserBehaviorConfig:
    """Configurable behavior parameters for the scripted user."""
    # Response timing
    response_gap_min_ms: float = 200.0
    response_gap_max_ms: float = 800.0
    response_gap_mean_ms: float = 400.0
    response_gap_std_ms: float = 150.0

    # Interruption
    interrupt_probability: float = 0.10
    interrupt_min_agent_duration_ms: float = 3000.0  # only interrupt if agent speaks > 3s

    # Clarification
    clarification_triggers: List[str] = field(default_factory=lambda: [
        "i'm not sure", "what do you mean", "can you explain",
    ])
    clarification_probability: float = 0.15

    # Correction
    correction_triggers: List[str] = field(default_factory=lambda: [
        "that's wrong", "no,", "actually,", "incorrect",
    ])
    correction_probability: float = 0.10

    # Turn duration
    user_turn_min_ms: float = 800.0
    user_turn_max_ms: float = 3000.0
    user_turn_mean_ms: float = 1500.0
    user_turn_std_ms: float = 500.0

    # Max turns before conversation ends
    max_user_turns: int = 10


# Pre-defined conversation scripts — each is a list of (agent_utterance, user_response) pairs.
# The scripted user follows these but adds timing variation and stochastic behaviors.
CONVERSATION_SCRIPTS = [
    # Script 0: Simple Q&A
    [
        ("Hello! How can I help you today?", "Hi, I have a question about my account."),
        ("Sure, what would you like to know?", "I'm not sure what my current balance is."),
        ("Let me check that for you.", "Thank you."),
        ("Your balance is $1,250.42.", "Oh, that's higher than I expected."),
        ("Is there anything else I can help with?", "No, that's all. Thanks!"),
    ],
    # Script 1: Problem-solving
    [
        ("I see you're having trouble with the device.", "Yes, it keeps making a weird noise."),
        ("When did this start?", "About two days ago."),
        ("Has anything changed recently?", "I dropped it last week but it was fine."),
        ("That could be the cause.", "What should I do?"),
        ("Try restarting it first.", "Okay, I'll try that."),
        ("Let me know if the noise continues.", "Will do, thanks."),
    ],
    # Script 2: Ambiguous / needs clarification
    [
        ("I can help with that request.", "I want to change my plan."),
        ("Which plan would you like to switch to?", "The cheaper one."),
        ("I'm not sure which plan you mean. Can you be more specific?", "The basic plan."),
        ("Got it. Switching you to the basic plan.", "Thanks."),
        ("The change will take effect next billing cycle.", "That's fine."),
    ],
    # Script 3: Correction scenario
    [
        ("Your appointment is at 3 PM tomorrow.", "Actually, I moved it to 4 PM."),
        ("Let me update that for you.", "Thanks."),
        ("Done. Anything else?", "No, that's all."),
    ],
    # Script 4: Longer conversation
    [
        ("Welcome! What brings you in today?", "I need help with my order."),
        ("What's the issue?", "I received the wrong item."),
        ("I'm sorry about that. What did you order?", "I ordered a blue shirt, size medium."),
        ("And what did you receive?", "A red shirt, size large."),
        ("That's wrong. Let me fix this for you.", "Thank you."),
        ("I'll send the correct item today.", "How long will shipping take?"),
        ("About 3-5 business days.", "That works."),
        ("Is there anything else?", "No, that's all I needed."),
        ("Have a great day!", "You too."),
    ],
]


class ScriptedUser:
    """Tier 1: Deterministic scripted user agent with stochastic behaviors."""

    def __init__(self, config: Optional[UserBehaviorConfig] = None, seed: int = 0):
        self.config = config or UserBehaviorConfig()
        self.rng = random.Random(seed)
        self.state = UserState.SILENT
        self._script: List[tuple[str, str]] = []
        self._turn_idx: int = 0
        self._current_response: str = ""
        self._response_delay_ms: float = 0.0
        self._turn_duration_ms: float = 0.0
        self._speaking_start_ms: float = 0.0
        self._will_interrupt: bool = False
        self._will_clarify: bool = False
        self._will_correct: bool = False

    def reset(self, script_id: Optional[int] = None, seed: int = 0):
        """Reset for a new trial."""
        self.rng = random.Random(seed)
        self.state = UserState.SILENT
        self._turn_idx = 0
        self._current_response = ""
        self._response_delay_ms = 0.0
        self._turn_duration_ms = 0.0
        self._speaking_start_ms = 0.0
        self._will_interrupt = False
        self._will_clarify = False
        self._will_correct = False
        self._agent_speaking = False
        self._agent_just_finished = False

        if script_id is not None:
            self._script = CONVERSATION_SCRIPTS[script_id % len(CONVERSATION_SCRIPTS)]
        else:
            self._script = self.rng.choice(CONVERSATION_SCRIPTS)

    @property
    def agent_utterance(self) -> str:
        """Return the agent's next scripted utterance, or empty string."""
        if self._turn_idx < len(self._script):
            return self._script[self._turn_idx][0]
        return ""

    @property
    def user_response(self) -> str:
        """Return the user's current response, or empty string."""
        if self._turn_idx < len(self._script):
            return self._script[self._turn_idx][1]
        return ""

    @property
    def user_has_responded(self) -> bool:
        """Whether the user has started their response this turn."""
        return self._turn_idx > 0 and self.state != UserState.SILENT

    @property
    def done(self) -> bool:
        return self._turn_idx >= len(self._script)

    def on_agent_speech_start(self, text: str, time_ms: float):
        # Decide if user will interrupt
        self._will_interrupt = False
        self._will_clarify = False
        self._will_correct = False

        # Check for clarification triggers in agent text
        text_lower = text.lower()
        for trigger in self.config.clarification_triggers:
            if trigger in text_lower:
                if self.rng.random() < self.config.clarification_probability:
                    self._will_clarify = True
                break

    def on_agent_speech_end(self, text: str, duration_ms: float, time_ms: float):
        """Called when the agent finishes speaking. Returns response delay."""
        # Decide if user will interrupt (only if agent spoke long enough)
        if duration_ms >= self.config.interrupt_min_agent_duration_ms:
            if self.rng.random() < self.config.interrupt_probability:
                self._will_interrupt = True

        # Compute response delay
        delay = self._sample_response_delay()
        self._response_delay_ms = delay
        self._turn_duration_ms = self._sample_turn_duration()
        return delay

    def on_agent_yield(self, time_ms: float):
        """Called when the agent yields (stops speaking due to interruption)."""
        # User takes over immediately
        self._response_delay_ms = 50.0  # quick takeover
        self._turn_duration_ms = self._sample_turn_duration()

    def get_perception(self, time_ms: float) -> Dict[str, Any]:
        """Get the user's current state for perception simulation."""
        return {
            "user_speaking": self.state == UserState.SPEAKING,
            "user_interrupting": self.state == UserState.INTERRUPTING,
            "user_text": self._current_response if self.state == UserState.SPEAKING else "",
        }

    def tick(self, time_ms: float, dt_ms: float) -> Optional[Dict[str, Any]]:
        """
        Advance user state machine. Returns an event dict if something happens.
        Events: "user_speech_start", "user_speech_end", "user_interrupt"
        """
        if self.done:
            return None

        prev_state = self.state

        # State machine
        if self.state == UserState.SILENT:
            if self._response_delay_ms > 0:
                self._response_delay_ms -= dt_ms
                if self._response_delay_ms <= 0:
                    # Start speaking
                    if self._will_interrupt:
                        self.state = UserState.INTERRUPTING
                        event_type = "user_interrupt"
                    else:
                        self.state = UserState.SPEAKING
                        event_type = "user_speech_start"
                    self._current_response = self.user_response
                    self._speaking_start_ms = time_ms
                    return {
                        "type": event_type,
                        "text": self._current_response,
                        "time_ms": time_ms,
                        "will_clarify": self._will_clarify,
                        "will_correct": self._will_correct,
                    }

        elif self.state in (UserState.SPEAKING, UserState.INTERRUPTING):
            elapsed = time_ms - self._speaking_start_ms
            if elapsed >= self._turn_duration_ms:
                # Stop speaking
                self.state = UserState.SILENT
                event = {
                    "type": "user_speech_end",
                    "text": self._current_response,
                    "time_ms": time_ms,
                    "duration_ms": elapsed,
                }
                self._turn_idx += 1
                return event

        return None

    def _sample_response_delay(self) -> float:
        delay = self.rng.gauss(
            self.config.response_gap_mean_ms,
            self.config.response_gap_std_ms,
        )
        return max(self.config.response_gap_min_ms, min(delay, self.config.response_gap_max_ms))

    def _sample_turn_duration(self) -> float:
        duration = self.rng.gauss(
            self.config.user_turn_mean_ms,
            self.config.user_turn_std_ms,
        )
        return max(self.config.user_turn_min_ms, min(duration, self.config.user_turn_max_ms))


# ── Deterministic Speaking Policy (Phase 2 placeholder) ─────────────────────

class DeterministicSpeakingPolicy:
    """
    Deterministic state machine for speaking decisions.

    State transitions (from spec):
      IDLE   → LISTEN  on user_speech_start
      LISTEN → SPEAK   on user_speech_end + silence > 300ms, if queued
      LISTEN → WAIT    otherwise
      LISTEN → YIELD   on user_interrupts_agent
      SPEAK  → YIELD   on BOTH frame (collision)
      SPEAK  → LISTEN  on utterance_complete + user_responds_within_2s
      SPEAK  → CLARIFY on utterance_complete + no_user_response_within_2s
      REPAIR → LISTEN  on repair_utterance_complete
    """

    def __init__(
        self,
        silence_threshold_ms: float = 200.0,
        user_response_timeout_ms: float = 2000.0,
        collision_overlap_threshold_ms: float = 200.0,
        repair_text: str = "Sorry, go ahead.",
        clarify_text: str = "Could you repeat that?",
    ):
        self.state = AgentState.IDLE
        self._queued_utterance: str = ""
        self._repair_utterance: str = ""
        self._clarify_utterance: str = ""
        self._speech_start_ms: float = 0.0
        self._speech_end_ms: float = 0.0
        self._last_user_text: str = ""
        self._last_agent_text: str = ""
        self._history: List[Dict[str, str]] = []
        self._last_spoken_script: str = ""
        self._overlap_start_ms: float = 0.0
        self._overlap_duration_ms: float = 0.0
        self._user_responded_after_speech: bool = False
        self._clarify_timeout_ms: float = 0.0

        # Config
        self.silence_threshold_ms = silence_threshold_ms
        self.user_response_timeout_ms = user_response_timeout_ms
        self.collision_overlap_threshold_ms = collision_overlap_threshold_ms
        self.repair_text = repair_text
        self.clarify_text = clarify_text

    def reset(self):
        self.state = AgentState.IDLE
        self._queued_utterance = ""
        self._repair_utterance = ""
        self._clarify_utterance = ""
        self._speech_start_ms = 0.0
        self._speech_end_ms = 0.0
        self._last_user_text = ""
        self._last_agent_text = ""
        self._history = []
        self._last_spoken_script = ""
        self._overlap_start_ms = 0.0
        self._overlap_duration_ms = 0.0
        self._user_responded_after_speech = False
        self._clarify_timeout_ms = 0.0

    def decide(
        self,
        perception: PerceptionFrame,
        user_info: Dict[str, Any],
        agent_script_line: str = "",
    ) -> tuple[Action, str]:
        """
        Given current perception, return (action, text_to_speak).
        Implements the full deterministic state machine.
        """
        # Track overlap duration for collision detection
        if perception.attribution == "both" and self.state == AgentState.SPEAK:
            if self._overlap_start_ms == 0.0:
                self._overlap_start_ms = perception.time_ms
            self._overlap_duration_ms = perception.time_ms - self._overlap_start_ms
        else:
            self._overlap_start_ms = 0.0
            self._overlap_duration_ms = 0.0

        # Track whether user responded after agent finished speaking
        if self.state == AgentState.LISTEN and user_info.get("user_speaking"):
            self._user_responded_after_speech = True

        # Only queue if the script line is new (not already spoken or currently queued)
        if agent_script_line and agent_script_line != self._queued_utterance and agent_script_line != self._last_spoken_script:
            self._queued_utterance = agent_script_line

        # ── IDLE: wait for user to speak first ──
        if self.state == AgentState.IDLE:
            if perception.attribution == "world":
                # User started speaking — transition to LISTEN
                self.state = AgentState.LISTEN
            elif perception.attribution == "silence" and self._queued_utterance:
                # No user speech yet, but we have something to say (opening)
                text = self._queued_utterance
                self._queued_utterance = ""
                self._last_spoken_script = text
                self.state = AgentState.SPEAK
                return (Action.SPEAK, text)
            return (Action.WAIT, "")

        # ── LISTEN: decide whether to speak, wait, or yield ──
        if self.state == AgentState.LISTEN:
            # User is still speaking — keep listening
            if perception.attribution == "world":
                return (Action.WAIT, "")

            # User interrupted (rare in LISTEN state, but handle it)
            if user_info.get("user_interrupting"):
                return (Action.YIELD, "")

            # Cancel clarify if user is currently speaking
            if self._clarify_utterance and user_info.get("user_speaking"):
                self._clarify_utterance = ""
            # Clarify timeout: agent finished speaking, user didn't respond in time
            elif self._clarify_utterance and perception.time_ms >= self._clarify_timeout_ms:
                text = self._clarify_utterance
                self._clarify_utterance = ""
                return (Action.CLARIFY, text)

            # User finished speaking — check silence gap
            if perception.attribution in ("silence", "world") and self._queued_utterance:
                if perception.silence_duration_ms > self.silence_threshold_ms:
                    text = self._queued_utterance
                    self._queued_utterance = ""
                    self._last_spoken_script = text
                    self.state = AgentState.SPEAK
                    self._user_responded_after_speech = False
                    return (Action.SPEAK, text)

            return (Action.WAIT, "")

        # ── SPEAK: monitor for collisions and completion ──
        if self.state == AgentState.SPEAK:
            # Collision detected
            if perception.attribution == "both":
                if self._overlap_duration_ms >= self.collision_overlap_threshold_ms:
                    # Significant overlap — yield and queue repair
                    self._repair_utterance = self.repair_text
                    self.state = AgentState.REPAIR
                    return (Action.YIELD, "")
                else:
                    # Brief overlap — just yield
                    return (Action.YIELD, "")

            return (Action.WAIT, "")

        # ── REPAIR: emit repair utterance then return to LISTEN ──
        if self.state == AgentState.REPAIR:
            if self._repair_utterance:
                text = self._repair_utterance
                self._repair_utterance = ""
                self.state = AgentState.LISTEN
                return (Action.REPAIR, text)
            self.state = AgentState.LISTEN
            return (Action.WAIT, "")

        return (Action.WAIT, "")

    def on_speech_start(self, text: str, time_ms: float):
        self.state = AgentState.SPEAK
        self._speech_start_ms = time_ms
        self._last_agent_text = text
        self._history.append({"speaker": "agent", "text": text})

    def on_speech_end(self, text: str, time_ms: float):
        self._speech_end_ms = time_ms
        self._queued_utterance = ""
        self.state = AgentState.LISTEN
        # Only set clarify timeout if user hasn't responded yet
        if not self._user_responded_after_speech:
            self._clarify_utterance = self.clarify_text
            self._clarify_timeout_ms = time_ms + self.user_response_timeout_ms
        self._user_responded_after_speech = False

    def on_yield(self, time_ms: float):
        self.state = AgentState.LISTEN
        self._overlap_start_ms = 0.0
        self._overlap_duration_ms = 0.0

    def queue_utterance(self, text: str):
        self._queued_utterance = text

    def on_user_text(self, text: str):
        self._last_user_text = text
        self._history.append({"speaker": "user", "text": text})
        # Cancel any pending clarify — user responded
        self._clarify_utterance = ""


# ── Prompt-Based Policy (Phase 3) ────────────────────────────────────────────

@dataclass
class PromptPolicyConfig:
    """Configuration for the prompt-based speaking policy."""
    llm_callable: Optional[Any] = None
    model_name: str = "mock"
    max_retries: int = 3
    timeout_s: float = 10.0
    temperature: float = 0.1


class PromptSpeakingPolicy:
    """
    LLM prompt-based speaking policy (Phase 3 calibration baseline).

    Receives the same perception context as the deterministic policy but
    delegates the decision to an LLM prompt. Logs explanations for analysis.

    This is a calibration baseline, not the product policy.
    """

    DECISION_PROMPT = """You are a speaking policy for a conversational AI agent.
Your job is to decide when the agent should speak, wait, clarify, repair, or yield
in a turn-taking conversation.

Rules:
- Speak only when it's the agent's turn (user has finished speaking + brief silence)
- Wait while the user is speaking
- Yield immediately if the user interrupts the agent mid-sentence
- Repair (say "Sorry, go ahead.") if the agent and user spoke simultaneously
- Clarify (ask user to repeat) if the agent spoke and the user didn't respond
- Keep responses natural and conversational

Current conversation state:
- Agent state: {state}
- Last user utterance: "{user_text}"
- Last agent utterance: "{agent_text}"
- Silence duration: {silence_ms}ms
- VAD energy: {energy_db} dB
- Attribution: {attribution}
- Has queued utterance: {has_queued}
- Recent history:
{history}

Available actions: speak, wait, clarify, repair, yield

What should the agent do next? Respond with:
ACTION: <one of: speak, wait, clarify, repair, yield>
EXPLANATION: <1-2 sentences explaining why>
TEXT: <the exact text to speak, or "none">"""

    def __init__(self, config: Optional[PromptPolicyConfig] = None):
        self.config = config or PromptPolicyConfig()
        self.state = AgentState.IDLE
        self._queued_utterance: str = ""
        self._last_user_text: str = ""
        self._last_agent_text: str = ""
        self._history: List[Dict[str, str]] = []
        self._last_spoken_script: str = ""
        self._log: List[Dict[str, Any]] = []

    def reset(self):
        self.state = AgentState.IDLE
        self._queued_utterance = ""
        self._last_user_text = ""
        self._last_agent_text = ""
        self._history = []
        self._last_spoken_script = ""
        self._log = []

    @property
    def decision_log(self) -> List[Dict[str, Any]]:
        return list(self._log)

    def decide(
        self,
        perception: PerceptionFrame,
        user_info: Dict[str, Any],
        agent_script_line: str = "",
    ) -> tuple[Action, str]:
        # Only queue if the script line is new
        if agent_script_line and agent_script_line != self._queued_utterance and agent_script_line != self._last_spoken_script:
            self._queued_utterance = agent_script_line

        context = self._build_context(perception, user_info, agent_script_line)
        action, text, explanation = self._query_llm(context)

        self._log.append({
            "time_ms": perception.time_ms,
            "state": self.state.value,
            "perception": {
                "attribution": perception.attribution,
                "silence_duration_ms": perception.silence_duration_ms,
                "vad_energy_db": round(perception.vad_energy_db, 1),
                "user_speaking": user_info.get("user_speaking", False),
            },
            "action": action.value,
            "text": text,
            "explanation": explanation,
            "model": self.config.model_name,
        })

        if action == Action.SPEAK and text:
            self.state = AgentState.SPEAK
            self._last_spoken_script = text
            self._queued_utterance = ""
        elif action == Action.YIELD:
            self.state = AgentState.LISTEN
        elif action == Action.REPAIR:
            self.state = AgentState.LISTEN

        return (action, text)

    def on_speech_start(self, text: str, time_ms: float):
        self.state = AgentState.SPEAK
        self._last_agent_text = text
        self._history.append({"speaker": "agent", "text": text})

    def on_speech_end(self, text: str, time_ms: float):
        self.state = AgentState.LISTEN
        self._queued_utterance = ""

    def on_yield(self, time_ms: float):
        self.state = AgentState.LISTEN

    def queue_utterance(self, text: str):
        self._queued_utterance = text

    def on_user_text(self, text: str):
        self._last_user_text = text
        self._history.append({"speaker": "user", "text": text})

    def _build_context(
        self,
        perception: PerceptionFrame,
        user_info: Dict[str, Any],
        agent_script_line: str,
    ) -> Dict[str, Any]:
        recent_history = self._history[-3:] if self._history else []
        history_lines = []
        for h in recent_history:
            history_lines.append(f'  {h["speaker"].capitalize()}: "{h["text"]}"')
        history_str = "\n".join(history_lines) if history_lines else "  (none)"

        return {
            "state": self.state.value,
            "user_text": self._last_user_text or "(none)",
            "agent_text": self._last_agent_text or "(none)",
            "silence_ms": int(perception.silence_duration_ms),
            "energy_db": round(perception.vad_energy_db, 1),
            "attribution": perception.attribution,
            "has_queued": str(bool(self._queued_utterance)),
            "history": history_str,
        }

    def _query_llm(self, context: Dict[str, Any]) -> tuple[Action, str, str]:
        if self.config.llm_callable is None:
            return self._deterministic_fallback(context)
        prompt = self.DECISION_PROMPT.format(**context)
        try:
            response = self.config.llm_callable(prompt)
            return self._parse_response(response)
        except Exception:
            return self._deterministic_fallback(context)

    def _parse_response(self, response: str) -> tuple[Action, str, str]:
        action = Action.WAIT
        text = ""
        explanation = ""
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                try:
                    action = Action(line.split(":", 1)[1].strip().lower())
                except ValueError:
                    action = Action.WAIT
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TEXT:"):
                text = line.split(":", 1)[1].strip()
                if text.lower() in ("none", "", "n/a"):
                    text = ""
        if action == Action.SPEAK and not text:
            action = Action.WAIT
        return (action, text, explanation)

    def _deterministic_fallback(self, context: Dict[str, Any]) -> tuple[Action, str, str]:
        state = self.state
        attribution = context["attribution"]
        silence_ms = context["silence_ms"]
        has_queued = context["has_queued"] == "True"
        queued_text = self._queued_utterance

        if state == AgentState.IDLE:
            if attribution == "world":
                self.state = AgentState.LISTEN
                return (Action.WAIT, "", "User started speaking")
            if attribution == "silence" and has_queued:
                return (Action.SPEAK, queued_text, "Opening the conversation")
            return (Action.WAIT, "", "Waiting for user")

        if state == AgentState.LISTEN:
            if attribution == "world":
                return (Action.WAIT, "", "User is speaking")
            if attribution in ("silence", "world") and has_queued:
                if silence_ms > 200:
                    return (Action.SPEAK, queued_text, f"User finished, silence={silence_ms}ms")
            return (Action.WAIT, "", f"Waiting, silence={silence_ms}ms")

        if state == AgentState.SPEAK:
            if attribution == "both":
                return (Action.YIELD, "", "Collision detected")
            return (Action.WAIT, "", "Still speaking")

        if state == AgentState.REPAIR:
            return (Action.REPAIR, "Sorry, go ahead.", "Repairing after collision")

        return (Action.WAIT, "", "Default wait")


# ── Simulator ────────────────────────────────────────────────────────────────

@dataclass
class SimulatorConfig:
    """Configuration for the speaking policy simulator."""
    num_trials: int = 50
    time_step_ms: float = 30.0       # simulation tick (matches VAD frame)
    max_trial_duration_ms: float = 60000.0  # 60s max per trial
    max_turns_per_trial: int = 20
    user_config: UserBehaviorConfig = field(default_factory=UserBehaviorConfig)
    verbose: bool = False
    output_path: Optional[str] = None
    seed: int = 42
    # Policy selection: "deterministic" or "prompt"
    policy_type: str = "deterministic"
    # Prompt policy config (only used when policy_type="prompt")
    prompt_config: PromptPolicyConfig = field(default_factory=PromptPolicyConfig)
    # Deterministic policy config
    det_silence_threshold_ms: float = 200.0
    det_user_response_timeout_ms: float = 2000.0


class SpeakingPolicySimulator:
    """Main simulator harness."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.perception = SimulatedPerception()
        self.user = ScriptedUser(config.user_config)
        self.metrics: List[TrialMetrics] = []
        # Create the appropriate policy
        if config.policy_type == "prompt":
            self.policy = PromptSpeakingPolicy(config.prompt_config)
        else:
            self.policy = DeterministicSpeakingPolicy(
                silence_threshold_ms=config.det_silence_threshold_ms,
                user_response_timeout_ms=config.det_user_response_timeout_ms,
            )

    def run_all(self) -> AggregatedMetrics:
        """Run all trials and return aggregated metrics."""
        self.metrics = []
        for trial_id in range(self.config.num_trials):
            trial_metrics = self.run_trial(trial_id)
            self.metrics.append(trial_metrics)
            if self.config.verbose:
                print(f"  Trial {trial_id}: {trial_metrics.total_turns} turns, "
                      f"interruptions={trial_metrics.interruptions}, "
                      f"continuation={trial_metrics.turn_continuation_rate:.2f}")

        return self._aggregate()

    def run_trial(self, trial_id: int) -> TrialMetrics:
        """Run a single trial (one conversation)."""
        seed = self.config.seed + trial_id
        self.perception.reset()
        self.policy.reset()
        self.user.reset(seed=seed)

        m = TrialMetrics(trial_id=trial_id)
        time_ms = 0.0
        turn_records: List[TurnRecord] = []
        last_speaker = ""
        last_turn_end_ms = 0.0
        agent_speaking = False
        user_speaking = False
        agent_speech_text = ""
        agent_speech_start_ms = 0.0
        collision_active = False

        # Queue the first agent utterance
        if not self.user.done:
            self.policy.queue_utterance(self.user.agent_utterance)

        while time_ms < self.config.max_trial_duration_ms and m.total_turns < self.config.max_turns_per_trial:
            # 1. Get user state
            user_event = self.user.tick(time_ms, self.config.time_step_ms)

            # Process user events
            if user_event:
                if user_event["type"] in ("user_speech_start", "user_interrupt"):
                    user_speaking = True
                    self.policy.on_user_text(user_event["text"])

                    if user_event["type"] == "user_interrupt" and agent_speaking:
                        m.interruptions += 1
                        # Agent should yield
                        self.policy.on_yield(time_ms)
                        agent_speaking = False
                        collision_active = False

                    turn_records.append(TurnRecord(
                        turn_idx=m.total_turns,
                        speaker="user",
                        action=user_event["type"],
                        text=user_event["text"],
                        start_ms=time_ms,
                        end_ms=time_ms,
                        was_interruption=(user_event["type"] == "user_interrupt"),
                        gap_before_ms=time_ms - last_turn_end_ms if last_turn_end_ms > 0 else 0,
                    ))
                    if turn_records[-1].gap_before_ms > 0:
                        m.response_gaps_ms.append(turn_records[-1].gap_before_ms)
                    last_speaker = "user"
                    m.total_turns += 1
                    m.user_turns += 1

                elif user_event["type"] == "user_speech_end":
                    user_speaking = False
                    if turn_records and turn_records[-1].speaker == "user":
                        turn_records[-1].end_ms = time_ms
                    # Queue the next agent utterance
                    if not self.user.done:
                        self.policy.queue_utterance(self.user.agent_utterance)

            # 2. Generate perception frame
            asr_text = ""
            asr_speaker = ""
            if user_speaking:
                asr_text = self.user._current_response
                asr_speaker = "user"
            elif agent_speaking:
                asr_speaker = "agent"

            frame = self.perception.frame(
                time_ms=time_ms,
                agent_speaking=agent_speaking,
                user_speaking=user_speaking,
                asr_text=asr_text,
                speaker=asr_speaker,
            )

            # 3. Detect collisions
            if frame.attribution == "both" and not collision_active:
                collision_active = True
                m.collisions += 1
            elif frame.attribution != "both":
                collision_active = False

            # 4. Policy decision — feed agent's next scripted utterance
            agent_script_line = self.user.agent_utterance if not self.user.done else ""
            user_info = self.user.get_perception(time_ms)
            action, text = self.policy.decide(frame, user_info, agent_script_line)

            # 5. Execute action
            if action == Action.SPEAK and text and not agent_speaking:
                agent_speaking = True
                agent_speech_text = text
                agent_speech_start_ms = time_ms
                self.policy.on_speech_start(text, time_ms)

                gap_ms = time_ms - last_turn_end_ms if last_turn_end_ms > 0 else 0.0
                turn_records.append(TurnRecord(
                    turn_idx=m.total_turns,
                    speaker="agent",
                    action=Action.SPEAK.value,
                    text=text,
                    start_ms=time_ms,
                    end_ms=time_ms,
                    gap_before_ms=gap_ms,
                ))
                last_speaker = "agent"
                m.total_turns += 1
                m.agent_turns += 1

                # Notify user
                self.user.on_agent_speech_start(text, time_ms)

            elif action == Action.YIELD and agent_speaking:
                agent_speaking = False
                self.policy.on_yield(time_ms)
                if turn_records and turn_records[-1].speaker == "agent":
                    turn_records[-1].end_ms = time_ms

            elif action == Action.REPAIR:
                # Repair utterance — agent speaks briefly then yields floor
                m.repairs_attempted += 1
                repair_text = text
                # Simulate repair speech (short ~1s)
                repair_duration_ms = 1000.0
                # The repair is "spoken" inline — we track it as a turn
                turn_records.append(TurnRecord(
                    turn_idx=m.total_turns,
                    speaker="agent",
                    action=Action.REPAIR.value,
                    text=repair_text,
                    start_ms=time_ms,
                    end_ms=time_ms + repair_duration_ms,
                    was_repair=True,
                    gap_before_ms=0,
                ))
                m.total_turns += 1
                m.agent_turns += 1
                last_turn_end_ms = time_ms + repair_duration_ms
                last_speaker = "agent"
                # After repair, user takes over — advance time past repair
                time_ms += repair_duration_ms

            elif action == Action.CLARIFY:
                m.clarifications += 1
                clarify_text = text
                # Simulate clarify speech
                clarify_duration_ms = 1200.0
                turn_records.append(TurnRecord(
                    turn_idx=m.total_turns,
                    speaker="agent",
                    action=Action.CLARIFY.value,
                    text=clarify_text,
                    start_ms=time_ms,
                    end_ms=time_ms + clarify_duration_ms,
                    was_clarification=True,
                    gap_before_ms=0,
                ))
                m.total_turns += 1
                m.agent_turns += 1
                last_turn_end_ms = time_ms + clarify_duration_ms
                last_speaker = "agent"
                time_ms += clarify_duration_ms

            # 6. Simulate agent speech duration (based on text length)
            if agent_speaking:
                # Rough estimate: ~150ms per word + 200ms overhead
                word_count = len(agent_speech_text.split()) if agent_speech_text else 1
                estimated_duration_ms = max(800, word_count * 150 + 200)
                elapsed = time_ms - agent_speech_start_ms

                if elapsed >= estimated_duration_ms:
                    agent_speaking = False
                    self.policy.on_speech_end(agent_speech_text, time_ms)

                    if turn_records and turn_records[-1].speaker == "agent":
                        turn_records[-1].end_ms = time_ms

                    # Notify user
                    response_delay = self.user.on_agent_speech_end(
                        agent_speech_text, estimated_duration_ms, time_ms
                    )

                    last_turn_end_ms = time_ms

                    # Smooth turn: agent finished speaking, no collision
                    if last_speaker == "agent" and not collision_active:
                        if turn_records and turn_records[-1].speaker == "agent":
                            gap = turn_records[-1].gap_before_ms
                            if 0 < gap < 2000:  # reasonable turn boundary
                                m.smooth_turns += 1

            # Count smooth user turns too
            if user_event and user_event["type"] == "user_speech_start":
                if not collision_active:
                    gap = time_ms - last_turn_end_ms if last_turn_end_ms > 0 else 0
                    if 0 < gap < 2000:
                        m.smooth_turns += 1

            # 7. Check if conversation is done
            if self.user.done and not agent_speaking and not user_speaking:
                break

            time_ms += self.config.time_step_ms

        # Finalize metrics
        m.duration_ms = time_ms
        m.completed = self.user.done
        m.turn_records = [
            {
                "turn_idx": r.turn_idx,
                "speaker": r.speaker,
                "action": r.action,
                "text": r.text,
                "start_ms": round(r.start_ms, 1),
                "end_ms": round(r.end_ms, 1),
                "was_interruption": r.was_interruption,
                "was_collision": r.was_collision,
                "was_repair": r.was_repair,
                "gap_before_ms": round(r.gap_before_ms, 1),
            }
            for r in turn_records
        ]

        return m

    def _aggregate(self) -> AggregatedMetrics:
        """Aggregate metrics across all trials."""
        completed = [m for m in self.metrics if m.completed]
        n = len(self.metrics)

        def _mean(values: List[float]) -> float:
            return sum(values) / max(len(values), 1)

        def _std(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            mean = _mean(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            return variance ** 0.5

        ir = [float(m.interruption_rate) for m in completed] if completed else [0.0]
        tcr = [float(m.turn_continuation_rate) for m in completed] if completed else [0.0]

        return AggregatedMetrics(
            num_trials=n,
            completed_trials=len(completed),
            avg_interruption_rate=_mean([m.interruption_rate for m in completed]) if completed else 0,
            avg_turn_continuation_rate=_mean(tcr),
            avg_repair_success_rate=_mean([m.repair_success_rate for m in completed]) if completed else 0,
            avg_clarification_efficiency=_mean([m.clarification_efficiency for m in completed]) if completed else 0,
            avg_collision_rate=_mean([m.collision_rate for m in completed]) if completed else 0,
            avg_response_gap_ms=_mean([m.avg_response_gap_ms for m in completed]) if completed else 0,
            avg_duration_ms=_mean([m.duration_ms for m in completed]) if completed else 0,
            interruption_rate_std=_std(ir),
            turn_continuation_rate_std=_std(tcr),
            per_trial=[m.to_dict() for m in self.metrics],
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M1 Speaking Policy Simulator"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print per-trial output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-duration", type=int, default=60000,
                        help="Max trial duration in ms")
    parser.add_argument("--policy", type=str, default="deterministic",
                        choices=["deterministic", "prompt"],
                        help="Policy type: deterministic or prompt")
    args = parser.parse_args()

    config = SimulatorConfig(
        num_trials=args.trials,
        verbose=args.verbose,
        output_path=args.output,
        seed=args.seed,
        max_trial_duration_ms=float(args.max_duration),
        policy_type=args.policy,
    )

    print(f"M1 Speaking Policy Simulator — {args.policy} policy")
    print(f"  Trials: {config.num_trials}")
    print(f"  Seed: {config.seed}")
    print(f"  Max duration: {config.max_trial_duration_ms:.0f}ms per trial")
    print()

    sim = SpeakingPolicySimulator(config)
    agg = sim.run_all()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results ({agg.completed_trials}/{agg.num_trials} trials completed)")
    print(f"{'='*60}")
    print(f"  Interruption rate:        {agg.avg_interruption_rate:.4f} "
          f"(std={agg.interruption_rate_std:.4f})  [target < 0.15]")
    print(f"  Turn continuation rate:   {agg.avg_turn_continuation_rate:.4f} "
          f"(std={agg.turn_continuation_rate_std:.4f})  [target > 0.70]")
    print(f"  Repair success rate:      {agg.avg_repair_success_rate:.4f}  [target > 0.80]")
    print(f"  Clarification efficiency: {agg.avg_clarification_efficiency:.4f}  [target > 0.60]")
    print(f"  Collision rate:           {agg.avg_collision_rate:.4f}  [target < 0.10]")
    print(f"  Avg response gap:         {agg.avg_response_gap_ms:.1f}ms  [target 300-800ms]")
    print(f"  Avg trial duration:       {agg.avg_duration_ms:.1f}ms")
    print()

    # Check targets
    targets_met = 0
    checks = [
        ("Interruption rate", agg.avg_interruption_rate < 0.15),
        ("Turn continuation rate", agg.avg_turn_continuation_rate > 0.70),
        ("Repair success rate", agg.avg_repair_success_rate > 0.80),
        ("Clarification efficiency", agg.avg_clarification_efficiency > 0.60),
        ("Collision rate", agg.avg_collision_rate < 0.10),
        ("Response gap (lower)", agg.avg_response_gap_ms >= 300),
        ("Response gap (upper)", agg.avg_response_gap_ms <= 800),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        targets_met += int(passed)
        print(f"  [{status}] {name}")

    print(f"\n  Targets met: {targets_met}/{len(checks)}")

    # Write output
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "results", "m1_phase1_baseline.json"
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(agg.to_dict(), f, indent=2)
    print(f"\n  Results written to: {output_path}")


if __name__ == "__main__":
    main()
