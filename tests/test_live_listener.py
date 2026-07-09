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
    # Imitation target is the PLAYED sound: re-analyzed formants + prediction error
    # vs the perceived human vowel are recorded.
    assert trial["played_formants_hz"] is not None
    assert isinstance(trial["imitation_prediction_error_hz"], list)
    assert len(trial["imitation_prediction_error_hz"]) == 3


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


def test_live_listener_world_and_imitation_devices_split() -> None:
    # The world/test stimulus plays on the External Headphones device (2); the
    # imitation efference plays on the Mac mini Speakers device (3). They must be
    # independently configurable and not collide.
    from cora_infans.io.live_listener import LiveListener

    lst = LiveListener()
    assert lst.imitation_device_index == 3
    assert lst.stimulus.output_device_index == 2
    # Override both independently.
    lst2 = LiveListener(output_device_index=9, imitation_device_index=7)
    assert lst2.stimulus.output_device_index == 9
    assert lst2.imitation_device_index == 7


def test_play_efference_uses_imitation_device_index(monkeypatch) -> None:
    # The gated playback helper must route the efference to imitation_device_index
    # (not the world/test stimulus device).
    import cora_infans.io.live_listener as ll

    calls = []

    monkeypatch.setattr(
        ll,
        "play_wav_to_device",
        lambda path, *, device_index, sample_rate, volume_db: (  # type: ignore[misc]
            calls.append(device_index) or {"backend": "fake"}
        ),
    )
    lst = ll.LiveListener()
    out = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav", gate_fn=lambda: {"verdict": "promote_simulation_only"}
    )
    assert out["played"] is True
    assert calls == [3]
    # Constructor override is honored.
    calls.clear()
    ll.LiveListener(imitation_device_index=5).play_efference_if_authorized(
        wav_path="/tmp/x.wav", gate_fn=lambda: {"verdict": "promote_simulation_only"}
    )
    assert calls == [5]


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


def test_corollary_frame_tags_anchor_self_vs_world() -> None:
    # Frame tags classify the captured corollary window: the played-audio window
    # is tagged self (expected_self_audio), anchored at the measured latency.
    from cora_infans.io.live_listener import LiveListener

    tags = LiveListener().corollary_frame_tags(latency_ms=0.8, capture_ms=400)
    states = [t["speak_state"] for t in tags]
    assert states[0] == "READY_TO_SPEAK"
    assert "SPEAKING" in states
    speaking = next(t for t in tags if t["speak_state"] == "SPEAKING")
    assert speaking["expected_self_audio"] is True
    assert speaking["latency_aligned_ms"] == 0.8
    assert all(t["source"] == "corollary_feedback" for t in tags)


def test_forward_model_fit_recovers_self_gain(tmp_path) -> None:
    # Phase 2 Task 1 (measurement-only): given a real efference + an observed
    # mixture = self (delayed/attenuated efference) + a world sine, the grid fit
    # recovers an attenuation near the true value and reports a positive SVSR.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm

    sr = 48000
    n = sr // 4  # 250 ms
    eff = (np.sin(np.arange(n) * 0.05) * 0.8).astype(np.float32)
    world = (np.sin(np.arange(n) * 0.20) * 0.3).astype(np.float32)
    self_pcm = predict_self_pcm(eff, sample_rate=sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22)
    observed = mix_observed(self_pcm, world, world_gain=1.0)

    fit = LiveListener()._fit_self_forward_model(
        efference_pcm=eff, observed_pcm=observed, sample_rate=sr, latency_anchor_ms=-0.62
    )
    learned = fit["learned_params"]
    assert learned is not None
    # The toy model's attenuation is the dominant learnable gain; the fit should
    # land within the grid resolution of the true 0.55 (grid: 0.3,0.45,0.55,...).
    assert abs(learned["attenuation"] - 0.55) <= 0.1
    assert fit["svsr_db"] is not None and float(fit["svsr_db"]) > 0.0
    # Delay grid is centered on the anchor; recovered delay must be a grid point.
    assert learned["delay_ms"] in (-2.62, -0.62, 1.38, 3.38)


def test_motor_imitation_trial_fits_forward_model_when_captured(tmp_path) -> None:
    # With a capture_fn injected (real-io), the trial captures the corollary
    # window and records a forward_model_fit (measurement-only in Task 1).
    import numpy as np

    from cora_infans.io.live_listener import LiveListener

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")

    captured_paths = []

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms):
        captured_paths.append(path)
        # Mimic real capture_to_wav: the trial reads this file back, so it must
        # actually exist with valid PCM (silence is enough for the fit path).
        from cora_infans.io.motor_plan import write_wav

        write_wav(path, np.zeros(sample_rate * duration_ms // 1000, dtype=np.float32), sample_rate)
        return {"backend": "fake", "completed": True}

    trial = lst._motor_imitation_trial(
        pcm,
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=0,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        latency_anchor_ms=-0.62,
    )
    assert trial is not None
    assert trial["played"] is True
    assert captured_paths  # corollary window was captured
    assert "corollary_capture" in trial
    # Measurement-only: the fit is recorded; subtraction has not been wired yet.
    assert "forward_model_fit" in trial
    fit = trial["forward_model_fit"]
    assert fit is not None
    assert "svsr_db" in fit and "learned_params" in fit


def test_listen_live_motor_imitation_loop_closes_end_to_end(tmp_path) -> None:
    # Full loop without hardware: heard frame -> perception -> motor plan ->
    # gated playback -> corollary capture -> frame tags -> imitation quality.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.pipeline import AudioFrame

    # Speech-like frame: a 200ms @ 16kHz tone the analyzer can resolve, with a
    # channels signal + a captured PCM attached via last_pcm.
    pcm = (np.sin(np.arange(3200) * 0.05) * 30000).astype("<i2")

    class _FakeArray:
        channel_count = 2

        def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
            return AudioFrame(
                timestamp_ms=0, channel_energies=[0.05, 0.01], dominant_frequency_hz=1000.0
            )

    # last_pcm is read by the motor loop to analyze the heard waveform.
    _FakeArray.last_pcm = pcm  # type: ignore[attr-defined]

    class _FakeTuner:
        is_real_hardware = False

        def read_selected_azimuths(self):
            return [0.0, 90.0]

        def read_doa(self):
            class _D:
                azimuth_deg = 0.0
                speech_detected = True

            return _D()

    played_paths = []
    captured_paths = []

    def fake_play(path, *, device_index, sample_rate, volume_db):
        played_paths.append(path)
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms):
        captured_paths.append(path)
        return {"backend": "fake", "completed": True}

    listener = LiveListener(_FakeArray(), tuner=_FakeTuner())
    report = listener.listen(
        duration_ms=200,
        frame_ms=200,
        with_attention=True,
        with_motor_imitation=True,
        motor_imitation_gate_fn=lambda: {"verdict": "promote_simulation_only"},
        motor_imitation_playback_fn=fake_play,
        motor_imitation_out_dir=str(tmp_path),
        motor_imitation_volume_db=12.0,
        latency_anchor_ms=0.8,
    )

    mi = report["motor_imitation"]
    assert mi is not None
    assert mi["enabled"] is True
    assert mi["trial_count"] == 1
    trial = mi["trials"][0]
    # Loop closed: perceived -> played -> measured against played sound.
    assert trial["played"] is True
    assert trial["played_formants_hz"] is not None
    assert trial["imitation_prediction_error_hz"] is not None
    assert played_paths == [trial["efference_wav"]]
    # Latency anchor flows into the frame tags for the corollary window.
    tags = listener.corollary_frame_tags(latency_ms=0.8)
    assert any(t["speak_state"] == "SPEAKING" and t["latency_aligned_ms"] == 0.8 for t in tags)


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
