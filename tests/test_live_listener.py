"""Tests for the continuous live listener and parentese stimulus.

Deterministic tests use a fake mic array so they run without hardware. The real
listener test is skipped unless the XVF3800 is present (no synthetic fallback).
"""

import shutil

import numpy as np
import pytest
from pathlib import Path

from cora_infans.io.live_listener import LiveListener, ParenteseStimulus
from cora_infans.io.live_mic import LiveMicError
from cora_infans.pipeline import AudioFrame


class FakeArray:
    """Deterministic stand-in for LiveMicArray; yields scripted frames."""

    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)
        self._idx = 0
        self.channel_count = 2

    def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
        frame = self._frames[self._idx]
        self._idx = (self._idx + 1) % len(self._frames)
        return frame


def _frame(energies, hz=1000.0):
    return AudioFrame(timestamp_ms=0, channel_energies=energies, dominant_frequency_hz=hz)


def test_play_efference_requires_passed_gate() -> None:
    # The imitation playback helper must refuse (zero playback) unless the
    # readiness gate is promoted; this is the same gating discipline as the
    # SPEAK smoke. We inject a fake playback fn to assert it is never called.
    import cora_infans.io.live_listener as ll

    called = []

    def fake_play(path, *, device_index, sample_rate, volume_db):
        called.append(path)
        return {"backend": "fake", "completed": True}

    ll.play_wav_to_device = fake_play  # type: ignore[attr-defined]
    lst = LiveListener()
    refused = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "hold", "live_playback_allowed": False},
    )
    assert refused["played"] is False
    assert refused["reason"] == "gate_not_promoted"
    assert called == []

    ok = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
    )
    assert ok["played"] is True
    assert ok["reason"] == "authorized"
    assert called == ["/tmp/x.wav"]


def test_motor_imitation_trial_plays_when_authorized(tmp_path) -> None:
    # When play is requested and the gate is promoted, the re-synthesized
    # imitation is written to a WAV and the injected playback fn is called.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    played = {}

    def fake_play(path, *, device_index, sample_rate, volume_db):
        played["path"] = path
        return {"backend": "fake", "completed": True}

    trial = lst._motor_imitation_trial(
        pcm,
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=0,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        out_dir=str(tmp_path),
    )
    assert trial is not None
    assert trial["played"] is True
    assert trial["efference_wav"] == played["path"]
    assert Path(played["path"]).exists()


def test_motor_imitation_trial_no_play_when_gate_refused(tmp_path) -> None:
    # Gated off -> no playback, but analysis still happens.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    called = []

    def fake_play(path, *, device_index, sample_rate, volume_db):
        called.append(path)
        return {"backend": "fake", "completed": True}

    trial = lst._motor_imitation_trial(
        pcm,
        0.0,
        0,
        0,
        play=True,
        gate_fn=lambda: {"verdict": "hold"},
        playback_fn=fake_play,
        out_dir=str(tmp_path),
    )
    assert trial is not None
    assert trial["played"] is False
    assert called == []
    assert "perceived_nearest_vowel" in trial


def test_parentese_stimulus_synthesizes_nonempty_wav(tmp_path) -> None:
    stim = ParenteseStimulus()
    out = tmp_path / "stim.wav"
    stim.synthesize_wav(str(out), sweeps=3, sweep_ms=200)
    assert out.exists()
    import wave

    with wave.open(str(out), "rb") as w:
        assert w.getnframes() > 0
        # Pitch actually modulates: the synthesized signal has non-zero bandwidth.
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
        assert np.std(data) > 0.0


def test_corollary_capture_records_latency_anchor(tmp_path) -> None:
    # The corollary-feedback capture records the measured latency anchor and
    # delegates to the injected capture fn (real ffmpeg otherwise).
    from cora_infans.io.live_listener import LiveListener

    captured = {}
    calls = []

    def fake_capture(path, *, device, sample_rate, duration_ms):
        captured["path"] = path
        calls.append(duration_ms)
        return {"backend": "fake", "completed": True}

    out = LiveListener().capture_corollary_window(
        out_dir=str(tmp_path),
        capture_fn=fake_capture,
        latency_ms=0.8,
        capture_ms=400,
    )
    assert out["captured"] is True
    assert out["latency_aligned_ms"] == 0.8
    assert out["captured_wav"] == captured["path"]
    assert calls == [400]


def test_listener_accumulates_salient_and_dropout_counts() -> None:
    # Script: salient, salient, dropout, salient -> 1 dropout, 1 recovery.
    frames = [
        _frame([0.05, 0.01]),  # salient (max 0.05 > 0.02)
        _frame([0.04, 0.01]),  # salient
        _frame([0.005, 0.003]),  # dropout (max 0.005 < 0.02)
        _frame([0.06, 0.01]),  # salient (recovery)
    ]
    listener = LiveListener(FakeArray(frames), saliency_threshold=0.02)
    report = listener.listen(duration_ms=1000, frame_ms=250)

    assert report["frame_count"] == 4
    assert report["salient_frame_count"] == 3
    assert report["non_salient_frame_count"] == 1
    assert report["dropout_count"] == 1
    assert report["recovery_count"] == 1
    assert report["focus_path"] == ["ch0", "ch0", "ch0"]
    assert report["with_stimulus"] is False


def test_listener_tags_stimulus_flag() -> None:
    frames = [_frame([0.05, 0.01])]
    listener = LiveListener(FakeArray(frames), saliency_threshold=0.02)
    # Stimulus playback will try to shell out; in the fake path we just assert
    # the flag is recorded on the report without requiring real audio.
    report = listener.listen(duration_ms=200, frame_ms=200, with_stimulus=True)
    assert report["with_stimulus"] is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_listener_runs_on_real_hardware_if_present() -> None:
    from cora_infans.io.live_mic import LiveMicArray

    try:
        array = LiveMicArray()
        _ = array.channel_count
    except LiveMicError:
        pytest.skip("XVF3800 not available on this host")

    listener = LiveListener(array, saliency_threshold=0.02)
    report = listener.listen(duration_ms=800, frame_ms=200)
    assert report["source"] == "live_xvf3800"
    assert report["frame_count"] >= 1
    # No fabricated events: salient count can be 0 in a quiet room.
    assert 0 <= report["salient_frame_count"] <= report["frame_count"]
