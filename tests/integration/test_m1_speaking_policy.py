#!/usr/bin/env python3
"""M1 — Acceptance tests for the speaking policy simulator harness."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

# Add scripts dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from m1_speaking_policy_sim import (
    Action,
    AgentState,
    AggregatedMetrics,
    DeterministicSpeakingPolicy,
    PerceptionFrame,
    ScriptedUser,
    SimulatedPerception,
    SpeakingPolicySimulator,
    UserBehaviorConfig,
    SimulatorConfig,
    UserState,
    CONVERSATION_SCRIPTS,
)


class TestSimulatedPerception:
    def test_silence_frame(self):
        p = SimulatedPerception()
        frame = p.frame(time_ms=0.0, agent_speaking=False, user_speaking=False)
        assert frame.attribution == "silence"
        assert frame.vad_energy_db < -40

    def test_self_frame(self):
        p = SimulatedPerception()
        frame = p.frame(time_ms=0.0, agent_speaking=True, user_speaking=False)
        assert frame.attribution == "self"

    def test_world_frame(self):
        p = SimulatedPerception()
        frame = p.frame(time_ms=0.0, agent_speaking=False, user_speaking=True)
        assert frame.attribution == "world"

    def test_both_frame(self):
        p = SimulatedPerception()
        frame = p.frame(time_ms=0.0, agent_speaking=True, user_speaking=True)
        assert frame.attribution == "both"

    def test_silence_duration_tracking(self):
        p = SimulatedPerception()
        p.frame(time_ms=0.0, agent_speaking=True, user_speaking=False)
        p.frame(time_ms=30.0, agent_speaking=False, user_speaking=False)
        frame2 = p.frame(time_ms=60.0, agent_speaking=False, user_speaking=False)
        assert frame2.silence_duration_ms == 30.0


class TestScriptedUser:
    def test_reset(self):
        user = ScriptedUser(seed=42)
        user.reset()
        assert user.state == UserState.SILENT
        assert not user.done
        assert len(user.agent_utterance) > 0

    def test_script_progression(self):
        user = ScriptedUser(seed=42)
        user.reset(script_id=0)
        first_agent = user.agent_utterance
        first_user = user.user_response
        assert len(first_agent) > 0
        assert len(first_user) > 0

    def test_done_after_all_turns(self):
        user = ScriptedUser(
            config=UserBehaviorConfig(
                response_gap_min_ms=10,
                response_gap_max_ms=20,
                user_turn_min_ms=10,
                user_turn_max_ms=20,
            ),
            seed=42,
        )
        user.reset(script_id=3)  # shortest script (3 turns)
        # Simulate agent driving the conversation
        time_ms = 0.0
        agent_speaking = False
        agent_text = ""
        agent_start = 0.0
        while not user.done and time_ms < 10000:
            # Agent speaks first utterance
            if not agent_speaking and user.state == UserState.SILENT:
                agent_speaking = True
                agent_text = user.agent_utterance
                agent_start = time_ms
            # Agent finishes speaking
            if agent_speaking:
                word_count = len(agent_text.split()) if agent_text else 1
                duration = max(100, word_count * 50 + 50)
                if time_ms - agent_start >= duration:
                    agent_speaking = False
                    user.on_agent_speech_end(agent_text, duration, time_ms)
            user.tick(time_ms, 30.0)
            time_ms += 30.0
        assert user.done


class TestDeterministicPolicy:
    def test_initial_state(self):
        policy = DeterministicSpeakingPolicy()
        assert policy.state == AgentState.IDLE

    def test_speak_on_silence_with_queued(self):
        policy = DeterministicSpeakingPolicy()
        policy.queue_utterance("Hello!")
        frame = PerceptionFrame(
            time_ms=0.0,
            agent_speaking=False,
            user_speaking=False,
            vad_energy_db=-50.0,
            silence_duration_ms=500.0,
            attribution="silence",
            asr_text="",
            speaker="",
        )
        action, text = policy.decide(frame, {}, "")
        assert action == Action.SPEAK
        assert text == "Hello!"

    def test_wait_on_silence_without_queued(self):
        policy = DeterministicSpeakingPolicy()
        frame = PerceptionFrame(
            time_ms=0.0,
            agent_speaking=False,
            user_speaking=False,
            vad_energy_db=-50.0,
            silence_duration_ms=500.0,
            attribution="silence",
            asr_text="",
            speaker="",
        )
        action, text = policy.decide(frame, {}, "")
        assert action == Action.WAIT

    def test_yield_on_both(self):
        policy = DeterministicSpeakingPolicy()
        policy.state = AgentState.SPEAK
        frame = PerceptionFrame(
            time_ms=0.0,
            agent_speaking=True,
            user_speaking=True,
            vad_energy_db=-20.0,
            silence_duration_ms=0.0,
            attribution="both",
            asr_text="",
            speaker="",
        )
        action, text = policy.decide(frame, {}, "")
        assert action == Action.YIELD

    def test_script_line_queues(self):
        policy = DeterministicSpeakingPolicy()
        frame = PerceptionFrame(
            time_ms=0.0,
            agent_speaking=False,
            user_speaking=False,
            vad_energy_db=-50.0,
            silence_duration_ms=500.0,
            attribution="silence",
            asr_text="",
            speaker="",
        )
        action, text = policy.decide(frame, {}, "Hello from script!")
        assert action == Action.SPEAK
        assert text == "Hello from script!"


class TestSimulator:
    def test_single_trial_completes(self):
        config = SimulatorConfig(num_trials=1, seed=42, verbose=False)
        sim = SpeakingPolicySimulator(config)
        metrics = sim.run_trial(0)
        assert metrics.completed
        assert metrics.total_turns > 0
        assert metrics.duration_ms > 0

    def test_all_trials_complete(self):
        config = SimulatorConfig(num_trials=5, seed=42, verbose=False)
        sim = SpeakingPolicySimulator(config)
        agg = sim.run_all()
        assert agg.completed_trials == 5
        assert agg.num_trials == 5

    def test_metrics_output(self):
        config = SimulatorConfig(num_trials=3, seed=42, verbose=False)
        sim = SpeakingPolicySimulator(config)
        agg = sim.run_all()
        d = agg.to_dict()
        assert "avg_interruption_rate" in d
        assert "avg_turn_continuation_rate" in d
        assert "avg_response_gap_ms" in d
        assert len(d["per_trial"]) == 3

    def test_response_gap_in_range(self):
        """Response gaps should be roughly 200-800ms given the default config."""
        config = SimulatorConfig(num_trials=10, seed=42, verbose=False)
        sim = SpeakingPolicySimulator(config)
        agg = sim.run_all()
        # With default config (mean=400ms, std=150ms), most gaps should be in range
        assert 200 < agg.avg_response_gap_ms < 800

    def test_no_interruptions_for_short_utterances(self):
        """Agent utterances are short (< 3s), so interruptions should be rare."""
        config = SimulatorConfig(num_trials=10, seed=42, verbose=False)
        sim = SpeakingPolicySimulator(config)
        agg = sim.run_all()
        # Interruptions only happen when agent speaks > 3s
        # Our script utterances are short, so this should be 0 or very low
        assert agg.avg_interruption_rate < 0.05


class TestConversationScripts:
    def test_all_scripts_non_empty(self):
        for i, script in enumerate(CONVERSATION_SCRIPTS):
            assert len(script) > 0, f"Script {i} is empty"
            for agent_text, user_text in script:
                assert len(agent_text) > 0, f"Script {i} has empty agent utterance"
                assert len(user_text) > 0, f"Script {i} has empty user response"

    def test_minimum_scripts(self):
        assert len(CONVERSATION_SCRIPTS) >= 3
