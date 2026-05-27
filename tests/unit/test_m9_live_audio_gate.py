"""Tests for the M9 live audio promotion gate scaffold."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("/Users/juicebox/Projects/cora/scripts/m9_live_audio_gate.py")

spec = importlib.util.spec_from_file_location("m9_live_audio_gate", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_make_target_pcm_returns_metadata_and_audio():
    pcm, meta = module.make_target_pcm("alpha", 220.0, "sine", 1.0)
    assert len(pcm) == 16000
    assert meta["word"] == "alpha"
    assert meta["sound_id"].startswith("sound::")
    assert meta["source"] in {"text_association_store", "oscillator"}


def test_parse_targets_defaults_to_live_set():
    targets = module.parse_targets(None)
    assert len(targets) >= 4
    assert targets[0][0] == "alpha"


def test_parse_targets_custom_spec():
    targets = module.parse_targets("alpha:220:sine,beta:247:triangle")
    assert targets == [("alpha", 220.0, "sine"), ("beta", 247.0, "triangle")]


def test_run_gate_dry_run_passes():
    result = module.run_gate(
        word="alpha",
        freq_hz=220.0,
        waveform="sine",
        duration_s=1.0,
        input_device=None,
        output_device=None,
        threshold=0.99,
        dry_run=True,
    )
    assert result["verdict"]["passed"] is True
    assert result["score"]["overall"] == 1.0
    assert result["target"]["word"] == "alpha"
    assert result["capture"]["input_device_name"] == "dry-run"


def test_run_gate_multi_target_dry_run_passes():
    rows = [
        module.run_gate(
            word=word,
            freq_hz=freq_hz,
            waveform=waveform,
            duration_s=1.0,
            input_device=None,
            output_device=None,
            threshold=0.99,
            dry_run=True,
        )
        for word, freq_hz, waveform in module.DEFAULT_TARGETS[:3]
    ]
    assert all(row["verdict"]["passed"] for row in rows)
    assert all(row["score"]["overall"] == 1.0 for row in rows)
