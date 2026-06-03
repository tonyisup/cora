"""M8 — Distilled speaking policy model.

Small MLP that imitates the heuristic speaking policy.
Same architecture pattern as M2 DistilledSpeakingPolicy.

Usage:
    from m8_policy.model import M8DistilledPolicy

    model = M8DistilledPolicy(hidden_dim=64, dropout=0.1)
    logits = model(features_tensor)
    action_idx = logits.argmax(dim=-1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Feature dimensions
# ConversationFeatures has ~20 fields, but we encode a compact subset:
# - conversation_state: 5 states (one-hot) → 5 dims
# - is_agent_speaking, is_user_speaking, is_agent_deferred: 3 bools → 3 dims
# - seconds_since_last_user_speech: 1 float (clamped, log-scaled)
# - seconds_since_last_agent_speech: 1 float (clamped, log-scaled)
# - seconds_agent_has_been_speaking: 1 float (clamped)
# - seconds_user_has_been_speaking: 1 float (clamped)
# - total_turns, repair_count, clarification_count, yield_count: 4 counts
# - just_interrupted, just_collided, just_deferred, just_resumed: 4 bools
# - has_queued_output: 1 bool
# Total: 5 + 3 + 1 + 1 + 1 + 1 + 4 + 4 + 1 = 21 dims
STATE_DIM = 5
INPUT_DIM = 21
NUM_ACTIONS = 5  # speak, wait, clarify, repair, yield

STATE_INDEX = {"idle": 0, "listening": 1, "speaking": 2, "repair": 3, "clarification": 4}
ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]


def encode_m8_features(features: dict) -> torch.Tensor:
    """Encode ConversationFeatures dict into a model input tensor.

    Returns a float32 tensor of shape (INPUT_DIM,).
    """
    # State one-hot (5 dims)
    state = features.get("conversation_state", "idle")
    state_idx = STATE_INDEX.get(state, 0)
    state_onehot = torch.zeros(STATE_DIM, dtype=torch.float32)
    state_onehot[state_idx] = 1.0

    # Bools (7 dims)
    bools = torch.tensor([
        float(features.get("is_agent_speaking", False)),
        float(features.get("is_user_speaking", False)),
        float(features.get("is_agent_deferred", False)),
        float(features.get("just_interrupted", False)),
        float(features.get("just_collided", False)),
        float(features.get("just_deferred", False)),
        float(features.get("just_resumed", False)),
    ], dtype=torch.float32)

    # Timing features (4 dims) — log-scaled and clamped
    def _time_feat(seconds: float) -> float:
        """Log-scale time: 0s→0, 1s→0.3, 2s→0.48, 5s→0.7, 10s→0.85, 999→1.0"""
        s = min(max(seconds, 0.0), 999.0)
        if s <= 0.0:
            return 0.0
        return min(math.log1p(s) / math.log1p(100.0), 1.0)

    import math
    timing = torch.tensor([
        _time_feat(features.get("seconds_since_last_user_speech", 999.0)),
        _time_feat(features.get("seconds_since_last_agent_speech", 999.0)),
        _time_feat(features.get("seconds_agent_has_been_speaking", 0.0)),
        _time_feat(features.get("seconds_user_has_been_speaking", 0.0)),
    ], dtype=torch.float32)

    # Counts (5 dims) — clamped and normalized
    counts = torch.tensor([
        min(float(features.get("total_turns", 0)), 20.0) / 20.0,
        min(float(features.get("repair_count", 0)), 10.0) / 10.0,
        min(float(features.get("clarification_count", 0)), 10.0) / 10.0,
        min(float(features.get("yield_count", 0)), 20.0) / 20.0,
        float(features.get("has_queued_output", False)),
    ], dtype=torch.float32)

    return torch.cat([state_onehot, bools, timing, counts])


class M8DistilledPolicy(nn.Module):
    """Small MLP for M8 speaking policy.

    Maps conversation features → action logits.
    Same architecture as M2 DistilledSpeakingPolicy for consistency.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64,
                 dropout: float = 0.1, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, features: dict) -> tuple[str, int]:
        """Predict action from a features dict.

        Returns
        -------
        (action_name, action_idx) : tuple[str, int]
        """
        x = encode_m8_features(features)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits = self.net(x)
                idx = logits.argmax().item()
        finally:
            if was_training:
                self.train()
        return ACTION_NAMES[idx], idx
