"""Unit tests for M8 distilled policy model and dataset."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from m8_policy.model import (
    M8DistilledPolicy,
    encode_m8_features,
    ACTION_NAMES,
    INPUT_DIM,
    NUM_ACTIONS,
    STATE_INDEX,
)


def make_features(
    conversation_state: str = "idle",
    is_agent_speaking: bool = False,
    is_user_speaking: bool = False,
    **overrides,
) -> dict:
    f = {
        "conversation_state": conversation_state,
        "is_agent_speaking": is_agent_speaking,
        "is_agent_deferred": False,
        "is_user_speaking": is_user_speaking,
        "was_user_interrupted": False,
        "seconds_since_last_user_speech": 999.0,
        "seconds_since_last_agent_speech": 999.0,
        "seconds_agent_has_been_speaking": 0.0,
        "seconds_user_has_been_speaking": 0.0,
        "total_turns": 0,
        "repair_count": 0,
        "clarification_count": 0,
        "yield_count": 0,
        "just_interrupted": False,
        "just_collided": False,
        "just_deferred": False,
        "just_resumed": False,
        "has_queued_output": False,
    }
    f.update(overrides)
    return f


class TestEncodeM8Features:
    def test_output_shape(self):
        f = make_features()
        x = encode_m8_features(f)
        assert x.shape == (INPUT_DIM,)
        assert x.dtype == torch.float32

    def test_idle_state_encoding(self):
        f = make_features(conversation_state="idle")
        x = encode_m8_features(f)
        assert x[0].item() == 1.0  # idle one-hot
        assert x[1].item() == 0.0  # listening
        assert x[2].item() == 0.0  # speaking

    def test_speaking_state_encoding(self):
        f = make_features(conversation_state="speaking", is_agent_speaking=True)
        x = encode_m8_features(f)
        assert x[2].item() == 1.0  # speaking one-hot
        assert x[5].item() == 1.0  # is_agent_speaking bool

    def test_repair_state_encoding(self):
        f = make_features(conversation_state="repair")
        x = encode_m8_features(f)
        assert x[3].item() == 1.0  # repair one-hot

    def test_listening_state_encoding(self):
        f = make_features(conversation_state="listening", is_user_speaking=True)
        x = encode_m8_features(f)
        assert x[1].item() == 1.0  # listening one-hot
        assert x[6].item() == 1.0  # is_user_speaking bool

    def test_flags_encoded(self):
        f = make_features(
            just_interrupted=True,
            just_collided=True,
            just_deferred=True,
            just_resumed=True,
        )
        x = encode_m8_features(f)
        # Layout: state(5) + bools(7: agent_speaking, user_speaking, deferred, interrupted,
        #                        collided, deferred_flag, resumed) + timing(4) + counts(5)
        # Flags are at indices 8-11 (bools[3:7])
        assert x[8].item() == 1.0   # just_interrupted (bools[3])
        assert x[9].item() == 1.0   # just_collided (bools[4])
        assert x[10].item() == 1.0  # just_deferred (bools[5])
        assert x[11].item() == 1.0  # just_resumed (bools[6])

    def test_timing_clamped(self):
        f = make_features(seconds_since_last_user_speech=10000.0)
        x = encode_m8_features(f)
        # Should be clamped to ~1.0 (log-scaled)
        assert x[11].item() <= 1.0


class TestM8DistilledPolicy:
    def test_init(self):
        model = M8DistilledPolicy()
        assert model is not None

    def test_forward_shape(self):
        model = M8DistilledPolicy()
        x = torch.randn(1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1, NUM_ACTIONS)

    def test_forward_batch(self):
        model = M8DistilledPolicy()
        x = torch.randn(32, INPUT_DIM)
        out = model(x)
        assert out.shape == (32, NUM_ACTIONS)

    def test_predict(self):
        model = M8DistilledPolicy()
        f = make_features(conversation_state="speaking", is_agent_speaking=True)
        action, idx = model.predict(f)
        assert action in ACTION_NAMES
        assert 0 <= idx < NUM_ACTIONS

    def test_predict_idle(self):
        model = M8DistilledPolicy()
        f = make_features(conversation_state="idle")
        action, idx = model.predict(f)
        assert action in ACTION_NAMES  # should be a valid action

    def test_parameter_count(self):
        model = M8DistilledPolicy()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # 21*64 + 64 + 64*64 + 64 + 64*5 + 5 = 1344+64+4096+64+320+5 = 5893
        assert n_params == 5893

    def test_save_load(self):
        model = M8DistilledPolicy()
        f = make_features(conversation_state="speaking", is_agent_speaking=True)
        action_before, _ = model.predict(f)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name

        try:
            torch.save({"model_state": model.state_dict(), "config": {
                "input_dim": INPUT_DIM, "hidden_dim": 64, "dropout": 0.1, "num_actions": NUM_ACTIONS
            }}, path)

            ckpt = torch.load(path, map_location="cpu")
            model2 = M8DistilledPolicy(
                input_dim=ckpt["config"]["input_dim"],
                hidden_dim=ckpt["config"]["hidden_dim"],
                dropout=ckpt["config"]["dropout"],
                num_actions=ckpt["config"]["num_actions"],
            )
            model2.load_state_dict(ckpt["model_state"])
            action_after, _ = model2.predict(f)

            assert action_before == action_after
        finally:
            os.unlink(path)

    def test_all_actions_reachable(self):
        """Ensure the model can output any action for some input."""
        model = M8DistilledPolicy()
        # Test with different feature combinations
        for state in STATE_INDEX:
            for speaking in [True, False]:
                f = make_features(conversation_state=state, is_agent_speaking=speaking)
                action, _ = model.predict(f)
                assert action in ACTION_NAMES
