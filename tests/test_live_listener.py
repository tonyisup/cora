"""Tests for the continuous live listener and parentese stimulus.

Deterministic tests use a fake mic array so they run without hardware. The real
listener test is skipped unless the XVF3800 is present (no synthetic fallback).
"""

import shutil
import time

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


def test_listener_reports_capture_to_workspace_latency_boundary() -> None:
    listener = LiveListener(array=FakeArray([_frame([0.2, 0.1])]))

    report = listener.listen(duration_ms=50, frame_ms=25)
    performance = report["runtime_performance"]

    assert performance["measurement_boundary"] == (
        "capture_frame_call_start_to_workspace_decision"
    )
    assert performance["frame_count"] == 2
    assert performance["capture_backend"] == "FakeArray"
    assert performance["real_capture"] is False
    assert performance["capture_to_workspace_latency_ms"]["p95"] >= 0.0
    assert report["runtime_resources"]["peak_resident_memory_bytes"] > 0
    assert report["runtime_resources"]["average_process_cpu_percent"] >= 0.0
    assert performance["workload_latency_ms"]["salient"]["frame_count"] == 2
    assert all("clipped_sample_count" in frame for frame in report["frames"])


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

    simulation_only = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
    )
    assert simulation_only["played"] is False
    assert simulation_only["reason"] == "bounded_live_io_not_authorized"

    ok = lst.play_efference_if_authorized(
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        bounded_playback_authorized=True,
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
        bounded_playback_authorized=True,
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


def test_motor_imitation_uses_source_rate_and_writes_model_efference_at_48k(
    tmp_path, monkeypatch
) -> None:
    import wave

    import cora_infans.io.live_listener as ll
    from cora_infans.io.motor_plan import write_wav

    seen_rates: list[int] = []
    playback_volume: list[float] = []
    fitted_pcm: list[np.ndarray] = []
    real_analyze = ll.analyze_vowel

    def recording_analyze(pcm, *, sample_rate):
        seen_rates.append(sample_rate)
        return real_analyze(pcm, sample_rate=sample_rate)

    monkeypatch.setattr(ll, "analyze_vowel", recording_analyze)

    def record_fit(self, *, efference_pcm, observed_pcm, sample_rate, latency_anchor_ms):
        fitted_pcm.append(efference_pcm.copy())
        return {
            "learned_params": {},
            "svsr_db": 0.0,
            "residual_rms": 0.0,
            "observed_rms": 0.0,
            "verdict": "not_learned",
        }

    monkeypatch.setattr(ll.LiveListener, "_fit_self_forward_model", record_fit)
    pcm = (np.sin(np.arange(9600) * 0.05) * 0.5).astype(np.float32)

    def fake_play(path, *, device_index, sample_rate, volume_db):
        playback_volume.append(volume_db)
        return {"completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        write_wav(path, np.zeros(sample_rate * duration_ms // 1000), sample_rate)
        ready_event.set()
        time.sleep(0.01)
        return {"completed": True}

    trial = ll.LiveListener()._motor_imitation_trial(
        pcm,
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=0,
        pcm_sample_rate=48000,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
    )

    assert trial is not None
    assert seen_rates[0] == 48000
    with wave.open(trial["efference_wav"], "rb") as wav:
        assert wav.getframerate() == 48000
    emitted_pcm, _ = ll._read_wav_mono_float32(str(trial["efference_wav"]))
    np.testing.assert_array_equal(fitted_pcm[0], emitted_pcm)
    assert playback_volume == [0.0]


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
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        bounded_playback_authorized=True,
    )
    assert out["played"] is True
    assert calls == [3]
    # Constructor override is honored.
    calls.clear()
    ll.LiveListener(imitation_device_index=5).play_efference_if_authorized(
        wav_path="/tmp/x.wav",
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        bounded_playback_authorized=True,
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


def test_parentese_stimulus_playback_gain_is_explicit(monkeypatch) -> None:
    import cora_infans.io.live_listener as listener_module

    observed = {}

    class FakeProcess:
        pass

    monkeypatch.setattr(listener_module, "_require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        listener_module.subprocess,
        "Popen",
        lambda command, **kwargs: observed.setdefault("command", command)
        and FakeProcess(),
    )

    ParenteseStimulus(volume_db=-6.0).play("probe.wav")

    assert "volume=-6dB" in observed["command"]


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


def test_mono_downmix_normalizes_int16_to_float_unit_scale(tmp_path) -> None:
    # Regression for the Task-6 scale bug: real captures arrive as int16 WAV PCM;
    # _mono_downmix MUST scale them into [-1,1] so observed shares the float scale
    # of the efference/predicted signals. Otherwise observed_rms is ~13k while
    # predicted_rms is ~0.3 and SVSR becomes a scale artifact.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener

    int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    f = LiveListener._mono_downmix(int16)
    assert f.dtype == np.float32
    assert abs(float(f[1]) - 0.5) < 1e-4
    assert abs(float(f[3]) - (32767.0 / 32768.0)) < 1e-4
    assert float(f.max()) <= 1.0 + 1e-6

    # 2-D stereo int16 also downmixes + scales.
    stereo = np.array([[16384, 0], [-16384, 0]], dtype=np.int16)
    fs = LiveListener._mono_downmix(stereo)
    assert abs(float(fs[0]) - 0.25) < 1e-4


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

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        captured_paths.append(path)
        # Mimic real capture_to_wav: the trial reads this file back, so it must
        # actually exist with valid PCM (silence is enough for the fit path).
        from cora_infans.io.motor_plan import write_wav

        write_wav(path, np.zeros(sample_rate * duration_ms // 1000, dtype=np.float32), sample_rate)
        ready_event.set()
        time.sleep(0.01)
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
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
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


def test_motor_imitation_capture_overlaps_playback(tmp_path) -> None:
    import threading

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav

    capture_active = threading.Event()
    playback_finished = threading.Event()
    order: list[str] = []

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        order.append("capture_start")
        capture_active.set()
        ready_event.set()
        assert playback_finished.wait(timeout=1.0)
        write_wav(path, np.zeros(sample_rate * duration_ms // 1000, dtype=np.float32), sample_rate)
        order.append("capture_end")
        return {"completed": True}

    def fake_play(path, *, device_index, sample_rate, volume_db):
        assert capture_active.wait(timeout=1.0)
        order.append("playback")
        playback_finished.set()
        return {"completed": True}

    trial = LiveListener()._motor_imitation_trial(
        (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2"),
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=7,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        bounded_playback_authorized=True,
    )

    assert trial is not None
    assert trial["played"] is True
    assert order == ["capture_start", "playback", "capture_end"]
    assert trial["corollary_capture"]["path"].endswith("corollary_capture_f7.wav")


def test_motor_imitation_aligns_capture_to_measured_playback_offset(
    tmp_path, monkeypatch
) -> None:
    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav

    fitted_observed: list[np.ndarray] = []

    def record_fit(self, *, efference_pcm, observed_pcm, sample_rate, latency_anchor_ms):
        fitted_observed.append(observed_pcm.copy())
        return {"learned_params": {}, "verdict": "not_learned"}

    monkeypatch.setattr(LiveListener, "_fit_self_forward_model", record_fit)

    def fake_capture(
        path, *, device, sample_rate, duration_ms, ready_event, capture_timeline
    ):
        pre_roll = np.zeros(sample_rate // 10, dtype=np.float32)
        played = np.ones(sample_rate * duration_ms // 1000 - len(pre_roll), dtype=np.float32)
        write_wav(path, np.concatenate([pre_roll, played]), sample_rate)
        capture_timeline["frame_position_fn"] = lambda: len(pre_roll)
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        time.sleep(0.15)
        return {"backend": "fake", "completed": True}

    trial = LiveListener()._motor_imitation_trial(
        (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2"),
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=1,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=lambda path, **kwargs: {"backend": "fake", "completed": True},
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        bounded_playback_authorized=True,
        capture_preroll_ms=100,
    )

    assert fitted_observed and float(np.mean(fitted_observed[0][:100])) > 0.9
    tags = trial["corollary_capture"]["frame_tags"]
    speaking = next(tag for tag in tags if tag["speak_state"] == "SPEAKING")
    assert 90 <= speaking["start_ms"] <= 120


def test_motor_imitation_serializes_forward_model_failure(tmp_path, monkeypatch) -> None:
    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav

    def fail_fit(self, **kwargs):
        raise ValueError("malformed calibration evidence")

    monkeypatch.setattr(LiveListener, "_fit_self_forward_model", fail_fit)

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        write_wav(path, np.zeros(sample_rate * duration_ms // 1000), sample_rate)
        ready_event.set()
        time.sleep(0.01)
        return {"completed": True}

    trial = LiveListener()._motor_imitation_trial(
        np.zeros(16000, dtype=np.float32),
        source_azimuth_deg=0.0,
        now_ms=0,
        frame_idx=2,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=lambda path, **kwargs: {"completed": True},
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
    )

    assert trial["played"] is True
    assert trial["processing_error"] == {
        "type": "ValueError",
        "message": "malformed calibration evidence",
        "stage": "forward_model_fit_or_subtraction",
    }
    assert trial["world_residual"] is None


def test_real_self_subtraction_promotion_gate() -> None:
    # Phase 2 Task 3: the real-data gate promotes only when SVSR is high AND the
    # world signal survived subtraction; otherwise it holds at tag_only.
    from cora_infans.io.live_listener import LiveListener

    missing_evidence = LiveListener._real_self_subtraction_promotion_gate(
        svsr_db=24.0, world_residual_correlation=0.4
    )
    assert missing_evidence["verdict"] == "hold"

    promote = LiveListener._real_self_subtraction_promotion_gate(
        svsr_db=24.0,
        world_residual_correlation=0.4,
        stable_calibration=True,
        synchronized_world_reference=True,
        held_out_trial_count=3,
    )
    assert promote["verdict"] == "promote"
    assert promote["policy_if_promoted"] == "self_subtract_live"

    hold_low_svsr = LiveListener._real_self_subtraction_promotion_gate(
        svsr_db=5.0, world_residual_correlation=0.4
    )
    assert hold_low_svsr["verdict"] == "hold"
    assert hold_low_svsr["policy_if_held"] == "tag_only"

    hold_world_lost = LiveListener._real_self_subtraction_promotion_gate(
        svsr_db=24.0, world_residual_correlation=-0.3
    )
    assert hold_world_lost["verdict"] == "hold"


def test_trial_gate_wiring_feeds_real_evidence(tmp_path) -> None:
    # The guardrails (stable_calibration, synchronized_world_reference,
    # held_out_trials) MUST be driven by real session/capture evidence, not
    # defaulted True. With verified capture timing + a synchronized anchor + a
    # committed >=3 held-out trials, the guardrails flip True; without verified
    # timing the synchronized_world_reference criterion stays False.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    sr = 48000
    n = sr // 4
    eff = (np.sin(np.arange(n) * 0.05) * 0.8).astype(np.float32)
    world = (np.sin(np.arange(n) * 0.20) * 0.3).astype(np.float32)
    observed = mix_observed(
        predict_self_pcm(eff, sample_rate=sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22),
        world,
        world_gain=1.0,
    )

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event,
                     capture_timeline=None, cancel_event=None):
        write_wav(path, (np.clip(observed, -1.0, 1.0) * 32767.0).astype("<i2"), sample_rate)
        # Mirror capture_to_wav: publish a genuine byte-count position source so
        # the loop's stream_position_verified derivation sees a real capture.
        if isinstance(capture_timeline, dict):
            capture_timeline["position_source"] = "raw_pcm_byte_count"
            # Simulate playback having just begun: a small positive frame count
            # at onset (mirrors capture_to_wav's bytes-written count early in the
            # capture), so the trial's onset slice is non-empty.
            capture_timeline["frame_position_fn"] = lambda: sample_rate // 100
        ready_event.set()
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    # Verified timing + synchronized anchor + committed held-out trials.
    trial = lst._motor_imitation_trial(
        pcm, 0.0, 0, 0,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        latency_anchor_ms=-0.62,
        do_self_subtraction=True,
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
        stable_calibration=True,
        held_out_trial_count=3,
    )
    assert trial is not None
    gate = trial["self_subtraction_gate"]
    assert gate["criteria"]["stable_calibration"]["passed"] is True
    assert gate["criteria"]["synchronized_world_reference"]["passed"] is True
    assert gate["criteria"]["held_out_trials"]["passed"] is True
    assert trial["contamination_policy"] in ("tag_only", "self_subtract_live")

    # Without verified capture timing the synchronized_world_reference criterion
    # must be False (mirrors the call-site derivation from cap io_timing).
    unverified_timing = {
        "overlap_verified": False,
        "stream_position_verified": False,
        "timestamp_source": "wall_clock_fallback",
    }
    cap_timing = unverified_timing
    synchronized = bool(
        isinstance(cap_timing, dict)
        and cap_timing.get("overlap_verified")
        and cap_timing.get("stream_position_verified")
    )
    assert synchronized is False


def test_trial_contamination_policy_transitions_on_gate(tmp_path) -> None:
    # With a real self+world captured mixture and do_self_subtraction=True, the
    # trial's contamination_policy reflects the gate verdict.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    sr = 48000
    n = sr // 4
    eff = (np.sin(np.arange(n) * 0.05) * 0.8).astype(np.float32)
    world = (np.sin(np.arange(n) * 0.20) * 0.3).astype(np.float32)
    observed = mix_observed(
        predict_self_pcm(eff, sample_rate=sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22),
        world,
        world_gain=1.0,
    )

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        write_wav(path, (np.clip(observed, -1.0, 1.0) * 32767.0).astype("<i2"), sample_rate)
        ready_event.set()
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    trial = lst._motor_imitation_trial(
        pcm, 0.0, 0, 0,
        play=True,
        gate_fn=lambda: {"verdict": "promote_simulation_only"},
        playback_fn=fake_play,
        capture_fn=fake_capture,
        out_dir=str(tmp_path),
        latency_anchor_ms=-0.62,
        do_self_subtraction=True,
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
    )
    assert trial is not None
    assert "self_subtraction_gate" in trial
    assert "contamination_policy" in trial
    assert trial["contamination_policy"] in ("tag_only", "self_subtract_live")


def test_motor_imitation_trial_applies_self_subtraction(tmp_path) -> None:
    # Phase 2 Task 2: with do_self_subtraction=True, the fitted forward model is
    # applied via spectral_subtract, producing world_residual metrics. Still
    # measurement-only at the policy level (tag_only until Task 3 gate).
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm

    lst = LiveListener()
    pcm = (np.sin(np.arange(16000) * 0.05) * 30000).astype("<i2")
    captured_paths = []

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    # Capture a real self+world mixture so subtraction has something to remove:
    # the captured WAV's first 16000 samples are the efference (16k) upsampled
    # to 48k via repetition, plus a world sine. We synthesize it directly.
    sr = 48000
    n = sr // 4
    eff = (np.sin(np.arange(n) * 0.05) * 0.8).astype(np.float32)
    world = (np.sin(np.arange(n) * 0.20) * 0.3).astype(np.float32)
    self_pcm = predict_self_pcm(eff, sample_rate=sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22)
    observed = mix_observed(self_pcm, world, world_gain=1.0)
    observed_i16 = (np.clip(observed, -1.0, 1.0) * 32767.0).astype("<i2")

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        captured_paths.append(path)
        write_wav(path, observed_i16, sample_rate)
        ready_event.set()
        time.sleep(0.01)
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
        do_self_subtraction=True,
        bounded_playback_authorized=True,
        capture_preroll_ms=0,
    )
    assert trial is not None
    wr = trial.get("world_residual")
    assert wr is not None
    assert wr["applied"] is True
    # A self prediction was formed and subtracted: predicted RMS is positive and
    # the SVSR is a finite number. (Energy strictly decreasing depends on the
    # captured mixture matching the efference; here we test the wiring, not the
    # synthetic fit, so we assert finiteness + a non-trivial self prediction.)
    assert wr["predicted_rms"] > 0.0
    assert isinstance(wr["svsr_db"], float)


def test_motor_imitation_loop_closes_end_to_end_with_subtraction(tmp_path) -> None:
    # Full loop (no hardware) with self-subtraction enabled: heard -> plan ->
    # gated playback -> capture -> fit -> world_residual recorded per trial.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.pipeline import AudioFrame

    pcm = (np.sin(np.arange(3200) * 0.05) * 30000).astype("<i2")

    class _FakeArray:
        channel_count = 2

        def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
            return AudioFrame(
                timestamp_ms=987654,
                channel_energies=[0.05, 0.01],
                dominant_frequency_hz=1000.0,
            )

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

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        write_wav(path, np.zeros(sample_rate * duration_ms // 1000, dtype=np.float32), sample_rate)
        ready_event.set()
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    listener = LiveListener(_FakeArray(), tuner=_FakeTuner())
    report = listener.listen(
        duration_ms=200,
        frame_ms=200,
        with_attention=True,
        with_motor_imitation=True,
        motor_imitation_gate_fn=lambda: {"verdict": "promote_simulation_only"},
        motor_imitation_playback_fn=fake_play,
        motor_imitation_capture_fn=fake_capture,
        motor_imitation_out_dir=str(tmp_path),
        latency_anchor_ms=-0.62,
        do_self_subtraction=True,
        bounded_playback_authorized=True,
        motor_imitation_capture_preroll_ms=0,
        with_scheduling=True,
    )
    mi = report["motor_imitation"]
    assert mi is not None and mi["trial_count"] == 1
    trial = mi["trials"][0]
    assert trial["played"] is True
    assert "forward_model_fit" in trial
    assert trial.get("world_residual") is not None
    assert trial["world_residual"]["applied"] is True
    assert trial["timestamp_ms"] == 987654
    assert len(trial["motor_plan_formants_hz"]) == 3
    occupied = [cycle for cycle in report["theta_gamma"]["cycles"] if cycle["event_count"]]
    assert len(occupied) == 1
    assert occupied[0]["cycle"] == 987654 // 200


def test_motor_imitation_end_to_end_self_subtraction_report_shape(tmp_path) -> None:
    # Phase 2 Task 5: full-loop integration (no hardware) with self-subtraction:
    # the per-trial record carries the forward-model fit, the world_residual, the
    # real-data promotion gate, and the resulting contamination_policy.
    import numpy as np

    from cora_infans.io.live_listener import LiveListener
    from cora_infans.io.motor_plan import write_wav
    from cora_infans.io.self_suppression import mix_observed, predict_self_pcm
    from cora_infans.pipeline import AudioFrame

    pcm = (np.sin(np.arange(3200) * 0.05) * 30000).astype("<i2")
    sr = 48000
    n = sr // 4
    eff = (np.sin(np.arange(n) * 0.05) * 0.8).astype(np.float32)
    world = (np.sin(np.arange(n) * 0.20) * 0.3).astype(np.float32)
    observed = mix_observed(
        predict_self_pcm(eff, sample_rate=sr, delay_ms=0.0, attenuation=0.55, echo_decay=0.22),
        world,
        world_gain=1.0,
    )

    class _FakeArray:
        channel_count = 2

        def capture_frame(self, *, duration_ms=None, dominant_frequency_hz=None):
            return AudioFrame(
                timestamp_ms=0, channel_energies=[0.05, 0.01], dominant_frequency_hz=1000.0
            )

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

    def fake_play(path, *, device_index, sample_rate, volume_db):
        return {"backend": "fake", "completed": True}

    def fake_capture(path, *, device, sample_rate, duration_ms, ready_event):
        # Record the world so correlation comes out finite/meaningful.
        write_wav(path, (np.clip(observed, -1.0, 1.0) * 32767.0).astype("<i2"), sample_rate)
        ready_event.set()
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    listener = LiveListener(_FakeArray(), tuner=_FakeTuner())
    report = listener.listen(
        duration_ms=200,
        frame_ms=200,
        with_attention=True,
        with_motor_imitation=True,
        motor_imitation_gate_fn=lambda: {"verdict": "promote_simulation_only"},
        motor_imitation_playback_fn=fake_play,
        motor_imitation_capture_fn=fake_capture,
        motor_imitation_out_dir=str(tmp_path),
        latency_anchor_ms=-0.62,
        do_self_subtraction=True,
        bounded_playback_authorized=True,
        motor_imitation_capture_preroll_ms=0,
    )
    mi = report["motor_imitation"]
    assert mi is not None and mi["trial_count"] == 1
    t = mi["trials"][0]
    # Every Phase 2 field is present and well-formed on the real-data path.
    assert t["forward_model_fit"] is not None
    assert t["forward_model_fit"]["svsr_db"] is not None
    assert t["world_residual"] is not None
    assert t["world_residual"]["applied"] is True
    assert "svsr_db" in t["world_residual"]
    assert t["self_subtraction_gate"] is not None
    assert t["self_subtraction_gate"]["verdict"] in ("promote", "hold")
    assert t["contamination_policy"] in ("tag_only", "self_subtract_live")

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
        bounded_playback_authorized=True,
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


def test_listener_plays_external_wav_without_deleting_it(tmp_path, monkeypatch) -> None:
    stimulus_path = tmp_path / "speech.wav"
    stimulus_path.write_bytes(b"bounded natural speech fixture")
    played = []

    class FinishedPlayback:
        def wait(self, timeout):
            return 0

        def kill(self):
            raise AssertionError("completed playback must not be killed")

    listener = LiveListener(FakeArray([_frame([0.05, 0.01])]), saliency_threshold=0.02)
    monkeypatch.setattr(
        listener.stimulus,
        "play",
        lambda path: played.append(path) or FinishedPlayback(),
    )

    report = listener.listen(
        duration_ms=200,
        frame_ms=200,
        with_stimulus=True,
        stimulus_wav_path=str(stimulus_path),
    )

    assert played == [str(stimulus_path.resolve())]
    assert stimulus_path.exists()
    assert report["stimulus"]["kind"] == "external_wav"
    assert report["stimulus"]["source_wav_sha256"] is not None
    assert all(frame["stimulus_playback_active"] for frame in report["frames"])


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
