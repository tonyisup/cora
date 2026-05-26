"""M8 — Scripted Episode Environment for PPO training.

Wraps the scripted conversation episodes in a gym-like reset()/step()
interface compatible with the PPO training loop. Each episode generates
observations in MultiStepTurnEnv format (state/attribution/silence/etc)
from scripted event timelines.

This bridges the PPO trainer (which expects MultiStepTurnEnv obs format)
with the scripted conversation simulator (which uses ConversationTracker).

Usage:
    from m8_scripted_env import ScriptedEpisodeEnv
    from m8_conversation_simulator import create_default_episodes

    episodes = create_default_episodes()
    env = ScriptedEpisodeEnv(episodes)

    obs = env.reset()
    next_obs, reward, done, info = env.step("speak")
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from m8_conversation_simulator import create_default_episodes


class ScriptedEpisodeEnv:
    """Gym-like environment that replays scripted conversational episodes.

    Each episode is a scripted timeline of agent output and user speech events.
    Observations are in MultiStepTurnEnv format:
        - state: str (idle/listen/speak/repair/clarify)
        - attribution: str (silence/self/world/both)
        - silence_duration_ms: float
        - has_queued_utterance: bool
        - user_speaking: bool
        - agent_speaking: bool
        - queue_depth: int
    """

    def __init__(self, episodes: Optional[List[Dict[str, Any]]] = None, seed: int = 42):
        self._episodes = episodes or create_default_episodes()
        self._rng = random.Random(seed)
        self._episode_idx = 0
        self._t = 0.0
        self._dt = 0.03
        self._episode: Optional[Dict[str, Any]] = None
        self._prev_features: Optional[Dict[str, Any]] = None

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def reset(self, episode_id: Optional[int] = None) -> Dict[str, Any]:
        """Reset to a new episode. Returns initial observation."""
        if episode_id is None:
            episode_id = self._episode_idx % len(self._episodes)
        self._episode_idx = episode_id + 1

        # Optionally perturb the episode for training diversity
        ep = self._episodes[episode_id % len(self._episodes)]
        self._episode = self._perturb(ep)
        self._t = 0.0

        obs = self._build_obs()
        self._prev_features = self._obs_to_conversation_features(obs)
        return obs

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance one time step with the given action.

        Returns (next_obs, reward, done, info).
        The reward is the M8-shaped reward for this transition.
        """
        from m8_reward import M8RewardConfig, compute_m8_reward
        reward_config = M8RewardConfig()

        self._t += self._dt
        next_obs = self._build_obs()

        # Compute reward from prev → current transition
        prev_feat = dict(self._prev_features) if self._prev_features else self._obs_to_conversation_features(next_obs)
        curr_feat = self._obs_to_conversation_features(next_obs)
        signal = compute_m8_reward(reward_config, action, prev_feat, curr_feat)

        done = self._t >= self._episode["duration_s"]

        info = {
            "episode_id": self._episode_idx - 1,
            "episode_name": self._episode["id"],
            "time_s": self._t,
            "duration_s": self._episode["duration_s"],
            "reward_reason": signal.reason,
            "reward_components": signal.components,
        }

        self._prev_features = curr_feat
        return next_obs, signal.reward, done, info

    def rollout(self, policy) -> List[Dict[str, Any]]:
        """Run one full rollout across all episodes with the given policy.

        Returns a list of (obs, action, reward, done, info) dicts.
        """
        steps = []
        for ep_id in range(len(self._episodes)):
            obs = self.reset(ep_id)
            done = False
            while not done:
                if hasattr(policy, "predict"):
                    action = policy.predict(obs)
                else:
                    action = policy(obs)
                next_obs, reward, done, info = self.step(action)
                steps.append({
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_obs,
                    "done": done,
                    "info": info,
                })
                obs = next_obs
        return steps

    def _build_obs(self) -> Dict[str, Any]:
        """Build observation from the current episode state at time self._t."""
        ep = self._episode
        agent_output = ep.get("agent_output", [])
        user_speech = ep.get("user_speech", [])

        # TTS state
        tts_playing = False
        tts_position_s = 0.0
        tts_total_duration_s = 0.0
        for ao in agent_output:
            if ao["time"] <= self._t < ao["time"] + ao["duration"]:
                tts_playing = True
                tts_position_s = self._t - ao["time"]
                tts_total_duration_s = ao["duration"]
                break

        # User speech state
        user_speaking = any(us["start"] <= self._t < us["end"] for us in user_speech)

        # Verdict
        if tts_playing and user_speaking:
            verdict = "BOTH"
        elif tts_playing:
            verdict = "SELF"
        elif user_speaking:
            verdict = "WORLD"
        else:
            verdict = "SILENCE"

        # Controller state (simplified)
        if tts_playing:
            controller_state = "SPEAK"
        elif user_speaking:
            controller_state = "LISTEN"
        else:
            controller_state = "IDLE"

        # Compute silence duration (time since last event)
        silence_ms = self._compute_silence_ms(user_speech, agent_output)

        # Queue depth
        queue_depth = 0
        for ao in agent_output:
            if self._t < ao["time"]:
                queue_depth += 1

        # Map controller state to M4-style state string
        state_map = {"IDLE": "idle", "LISTEN": "listen", "SPEAK": "speak"}

        return {
            "state": state_map.get(controller_state, "listen"),
            "attribution": {
                "SILENCE": "silence",
                "SELF": "self",
                "WORLD": "world",
                "BOTH": "both",
            }.get(verdict, "silence"),
            "silence_duration_ms": silence_ms,
            "has_queued_utterance": queue_depth > 0 or any(
                self._t < ao["time"] for ao in agent_output
            ),
            "user_speaking": user_speaking,
            "agent_speaking": tts_playing,
            "queue_depth": queue_depth,
        }

    def _compute_silence_ms(self, user_speech, agent_output) -> float:
        """Compute time since last acoustic event in ms."""
        last_event_time = 0.0
        for us in user_speech:
            if us["end"] <= self._t and us["end"] > last_event_time:
                last_event_time = us["end"]
            if us["start"] <= self._t and us["start"] > last_event_time:
                last_event_time = us["start"]
        for ao in agent_output:
            if ao["time"] + ao["duration"] <= self._t and ao["time"] + ao["duration"] > last_event_time:
                last_event_time = ao["time"] + ao["duration"]
        return (self._t - last_event_time) * 1000.0

    def _obs_to_conversation_features(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert env obs to ConversationFeatures format for M8 reward."""
        state_map = {
            "idle": "idle", "listen": "listening", "speak": "speaking",
            "repair": "repair", "clarify": "clarification",
        }
        silence_s = obs.get("silence_duration_ms", 0.0) / 1000.0
        return {
            "conversation_state": state_map.get(obs.get("state", "listen"), "listening"),
            "is_agent_speaking": obs.get("agent_speaking", False),
            "is_agent_deferred": obs.get("queue_depth", 0) > 0,
            "is_user_speaking": obs.get("user_speaking", False),
            "was_user_interrupted": obs.get("attribution", "") == "both" and obs.get("agent_speaking", False),
            "seconds_since_last_user_speech": silence_s,
            "seconds_since_last_agent_speech": 0.0,
            "seconds_agent_has_been_speaking": 0.0,
            "seconds_user_has_been_speaking": 0.0,
            "total_turns": 0,
            "repair_count": 0,
            "clarification_count": 0,
            "yield_count": 0,
            "just_interrupted": obs.get("attribution", "") == "both" and obs.get("agent_speaking", False),
            "just_collided": obs.get("attribution", "") == "both",
            "just_deferred": False,
            "just_resumed": False,
            "has_queued_utterance": obs.get("has_queued_utterance", False),
        }

    def _perturb(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Create a perturbed copy of an episode for training diversity."""
        ep = copy.deepcopy(episode)
        ep["id"] = f"{episode['id']}_perturbed"
        # Jitter event times by ±0.3s
        for ao in ep.get("agent_output", []):
            jitter = self._rng.uniform(-0.3, 0.3)
            ao["time"] = max(0.0, ao["time"] + jitter)
            dur_jitter = self._rng.uniform(-0.1, 0.1)
            ao["duration"] = max(0.5, ao["duration"] * (1.0 + dur_jitter))
        for us in ep.get("user_speech", []):
            jitter = self._rng.uniform(-0.2, 0.2)
            us["start"] = max(0.0, us["start"] + jitter)
            dur = us["end"] - us["start"]
            us["end"] = us["start"] + max(0.3, dur)
        return ep
