"""Fake-I/O SPEAK smoke harness before any real playback implementation."""

import wave
from pathlib import Path

import numpy as np

from cora_infans.live_speak_smoke import run_live_speak_smoke_trial


def _publish_capture_position(ready_event, capture_timeline, frames: int = 0) -> None:
    capture_timeline["frame_position_fn"] = lambda: frames
    capture_timeline["position_source"] = "synthetic_frame_count"
    ready_event.set()


def _fake_playback(path: str) -> dict[str, object]:
    assert Path(path).exists()
    return {"backend": "fake_playback", "started": True, "completed": True}


def _fake_capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
    samples = int(sample_rate * duration_ms / 1000)
    pcm = np.zeros(samples, dtype="<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return {"backend": "fake_capture", "frames": samples, "completed": True}


def test_live_speak_smoke_with_fake_io_writes_artifacts_and_tags_frames(tmp_path) -> None:
    report = run_live_speak_smoke_trial(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_fake_playback,
        capture_fn=_fake_capture,
        duration_ms=120,
        max_capture_ms=400,
    )

    assert report["mode"] == "bounded_live_speak_smoke"
    assert report["real_hardware"] is False
    assert report["live_playback_allowed"] is False
    assert report["contamination_policy"] == "tag_only"
    assert Path(report["artifacts"]["efference_wav"]).exists()
    assert Path(report["artifacts"]["captured_wav"]).exists()
    assert report["efference"]["vowel"] == "a"
    assert report["playback"]["backend"] == "fake_playback"
    assert report["capture"]["backend"] == "fake_capture"

    tags = report["frame_tags"]
    assert {tag["speak_state"] for tag in tags} >= {
        "READY_TO_SPEAK",
        "SPEAKING",
        "POST_SPEAK_SUPPRESSION",
        "RECOVERING",
    }
    assert all("efference_id" in tag for tag in tags)
    assert any(tag["expected_self_audio"] for tag in tags)
    assert all(tag["usable_for_world_learning"] is False for tag in tags if tag["expected_self_audio"])


def test_live_speak_smoke_requires_injected_io_by_default(tmp_path) -> None:
    report = run_live_speak_smoke_trial(vowel="a", out_dir=str(tmp_path), duration_ms=120)

    assert report["verdict"] == "blocked_missing_injected_io"
    assert report["real_hardware"] is False
    assert report["live_playback_allowed"] is False
    assert Path(report["artifacts"]["efference_wav"]).exists()
    assert report["artifacts"]["captured_wav"] is None


def test_recovering_frame_tag_has_positive_width_at_boundary(tmp_path) -> None:
    # At the boundary where capture does not exceed the speak+tail window,
    # RECOVERING must still occupy a real (positive-width) interval, not collapse
    # to zero length.
    report = run_live_speak_smoke_trial(
        vowel="a",
        out_dir=str(tmp_path),
        duration_ms=100,
        max_capture_ms=120,
    )

    tags = report["frame_tags"]
    recovering = [t for t in tags if t["speak_state"] == "RECOVERING"]
    assert recovering, "RECOVERING tag must be present"
    assert recovering[0]["end_ms"] > recovering[0]["start_ms"]


def test_latency_cross_correlation_recovers_known_delay() -> None:
    # Deterministic stub for the future real-hardware slice: a cross-correlation
    # estimator that recovers a known injected delay on synthetic PCM. This is
    # pure math, not live playback; the real-io path that would use it does not
    # exist yet.
    import numpy as np

    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    sample_rate = 16000
    ref = (np.sin(2 * np.pi * 0.01 * np.arange(2000)) * 30000).astype("<i2")
    delay_samples = 40
    captured = np.zeros(2000 + delay_samples, dtype="<i2")
    captured[delay_samples : delay_samples + 2000] = ref

    latency_ms = estimate_playback_latency_ms(
        efference_pcm=ref,
        captured_pcm=captured[:2000] if captured.size >= 2000 else captured,
        sample_rate=sample_rate,
    )

    assert latency_ms is not None
    assert abs(latency_ms - (delay_samples / sample_rate * 1000.0)) <= 2.0


def test_real_io_refuses_without_promoted_readiness_gate(tmp_path) -> None:
    # Without a promoted readiness gate, the real-I/O trial must abort BEFORE any
    # playback/capture, touching no hardware.
    from cora_infans.live_speak_smoke import run_real_io_speak_trial

    calls: list[str] = []

    def _play(path: str) -> dict[str, object]:
        calls.append("play")
        return {"backend": "fake", "completed": True}

    def _capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
        calls.append("capture")
        return {"backend": "fake", "completed": True}

    def _gate() -> dict[str, object]:
        return {"verdict": "hold", "live_playback_allowed": False}

    report = run_real_io_speak_trial(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_play,
        capture_fn=_capture,
        readiness_gate_fn=_gate,
    )

    assert report["verdict"] == "blocked_gates_not_promoted"
    assert report["real_hardware"] is False
    assert report["live_playback"] is False
    assert calls == []  # no I/O happened


def test_real_io_refuses_simulation_promotion_without_bounded_authorization(tmp_path) -> None:
    from cora_infans.live_speak_smoke import run_real_io_speak_trial

    calls: list[str] = []
    report = run_real_io_speak_trial(
        out_dir=str(tmp_path),
        playback_fn=lambda path: calls.append("play") or {"completed": True},
        capture_fn=lambda path, **kwargs: calls.append("capture") or {"completed": True},
        readiness_gate_fn=lambda: {
            "verdict": "promote_simulation_only",
            "live_playback_allowed": False,
        },
    )

    assert report["verdict"] == "blocked_bounded_live_io_not_authorized"
    assert report["live_playback"] is False
    assert calls == []


def test_real_io_rejects_silent_capture_as_unusable_measurement(tmp_path) -> None:
    # A completed I/O cycle is not a usable latency measurement when capture is silent.
    import threading

    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trial

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    playback_done = threading.Event()

    def _play(path: str) -> dict[str, object]:
        playback_done.set()
        return {"backend": "fake", "device_index": 2, "completed": True}

    def _capture(
        path: str, *, sample_rate: int, duration_ms: int, ready_event, capture_timeline
    ) -> dict[str, object]:
        import wave

        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        _publish_capture_position(ready_event, capture_timeline)
        assert playback_done.wait(timeout=1.0)
        return {"backend": "fake", "device": ":0", "frames": samples, "completed": True}

    report = run_real_io_speak_trial(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_play,
        capture_fn=_capture,
        readiness_gate_fn=_gate,
        sample_rate=16000,
        capture_preroll_ms=0,
        bounded_live_io_authorized=True,
    )

    assert report["verdict"] == "measurement_rejected"
    assert report["real_hardware"] is True
    assert report["live_playback"] is True
    assert report["live_playback_allowed"] is False
    assert report["artifacts"]["captured_wav"] is not None
    assert report["latency_aligned_ms"] is None
    assert report["usable"] is False
    # Frame tags present and ordered.
    states = [t["speak_state"] for t in report["frame_tags"]]
    assert states[0] == "READY_TO_SPEAK"
    assert "SPEAKING" in states


def test_real_io_trials_aggregate_latency_stats_with_stable_estimator(tmp_path) -> None:
    # Multi-trial harness: runs N real trials and reports latency mean/std/min/max.
    # With a deterministic injected estimator the spread is zero and the harness
    # flags latency_stable=True (the precondition for any later subtraction).
    import time

    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trials

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    def _fake_play(path: str) -> dict[str, object]:
        return {"backend": "fake", "completed": True}

    def _fake_capture(
        path: str, *, sample_rate: int, duration_ms: int, ready_event, capture_timeline
    ) -> dict[str, object]:
        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        _publish_capture_position(ready_event, capture_timeline)
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    def _est(*, efference_pcm: object, captured_pcm: object, sample_rate: int) -> float:
        return 12.0

    report = run_real_io_speak_trials(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_fake_play,
        capture_fn=_fake_capture,
        readiness_gate_fn=_gate,
        latency_estimator=_est,
        trials=4,
        sample_rate=16000,
        save_calibration=str(tmp_path / "cal.json"),
        capture_preroll_ms=0,
        bounded_live_io_authorized=True,
    )

    assert report["verdict"] == "trials_complete"
    assert report["trials_run"] == 4
    assert report["trials_ok"] == 4
    assert report["latency_stats"]["mean"] == 12.0
    assert report["latency_stats"]["std"] == 0.0
    assert report["latency_stats"]["min"] == 12.0
    assert report["latency_stats"]["max"] == 12.0
    assert report["latency_stable"] is True
    # The canonical tag set is published only when latency is stable, and it
    # anchors to the repeated stable mean (not any single noisy trial).
    assert report["latency_aligned_ms"] == 12.0
    tags = report["canonical_frame_tags"]  # type: ignore[assignment]
    assert isinstance(tags, list) and len(tags) == 4
    assert all(t["latency_aligned_ms"] == 12.0 for t in tags)  # type: ignore[index]
    assert all(t["latency_source"] == "stable_mean" for t in tags)  # type: ignore[index]
    assert all(t["latency_stable"] is True for t in tags)  # type: ignore[index]
    states = [t["speak_state"] for t in tags]  # type: ignore[index]
    assert states[0] == "READY_TO_SPEAK"
    assert "SPEAKING" in states
    # When a calibration path is requested and latency is stable, the artifact is
    # written and points back at the stable mean under the measurement conditions.
    assert report["calibration_path"] is not None
    import json as _json

    cal = _json.loads(open(report["calibration_path"]).read())  # type: ignore[arg-type]
    assert cal["kind"] == "playback_capture_loopback_latency"
    assert cal["latency_ms"] == 12.0
    assert cal["contamination_policy"] == "tag_only"


def test_real_io_trials_flag_unstable_when_spread_exceeds_threshold(tmp_path) -> None:
    # If the per-trial latency varies beyond the jitter threshold, the harness must
    # NOT claim stability — honest precondition for signal-level subtraction.
    import time

    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trials

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    def _fake_play(path: str) -> dict[str, object]:
        return {"backend": "fake", "completed": True}

    def _fake_capture(
        path: str, *, sample_rate: int, duration_ms: int, ready_event
    ) -> dict[str, object]:
        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        ready_event.set()
        time.sleep(0.01)
        return {"backend": "fake", "completed": True}

    _vals = {"n": 0}

    def _est(*, efference_pcm: object, captured_pcm: object, sample_rate: int) -> float:
        _vals["n"] += 1
        return float(_vals["n"] * 5.0)  # 5, 10, 15, 20 -> spread exceeds 2 ms

    report = run_real_io_speak_trials(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_fake_play,
        capture_fn=_fake_capture,
        readiness_gate_fn=_gate,
        latency_estimator=_est,
        trials=4,
        latency_jitter_ms=2.0,
        sample_rate=16000,
        capture_preroll_ms=0,
        bounded_live_io_authorized=True,
    )

    assert report["trials_ok"] == 4
    assert report["latency_stats"]["std"] > 2.0
    assert report["latency_stable"] is False


def test_real_io_trials_tolerate_one_device_startup_outlier(tmp_path) -> None:
    import time
    import wave

    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trials

    values = iter([347.3, 347.5, 343.6, 358.4, 346.9, 346.6])

    def capture(path, *, sample_rate, duration_ms, ready_event, **kwargs):
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(np.zeros(sample_rate, dtype="<i2").tobytes())
        ready_event.set()
        time.sleep(0.01)
        return {"completed": True}

    report = run_real_io_speak_trials(
        out_dir=str(tmp_path),
        playback_fn=lambda path: {"completed": True},
        capture_fn=capture,
        readiness_gate_fn=lambda: {"verdict": "promote_simulation_only"},
        latency_estimator=lambda **kwargs: next(values),
        trials=6,
        latency_jitter_ms=2.0,
        capture_preroll_ms=0,
        bounded_live_io_authorized=True,
    )

    assert report["latency_stable"] is True
    assert report["rejected_latency_outliers"] == 1
    assert report["latency_stats"]["n"] == 5


def test_chirp_latency_is_recovered_deterministically_at_clean_snr() -> None:
    # The chirp probe recovers a known injected delay exactly (no spread) when the
    # captured signal is clean — the precondition for a stable real-world latency.
    # Under heavy room-tone noise the estimator degrades for both stimuli (documented
    # empirically); a clean recovery here proves the estimator+probe path is correct.
    import numpy as np

    from cora_infans.io.motor_plan import synthesize_chirp
    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    sample_rate = 16000
    true_delay = 53  # samples
    sig = synthesize_chirp(duration_ms=300, sample_rate=sample_rate)
    ref = (sig * 30000).astype("<i2")
    captured = np.zeros(len(sig) + true_delay, dtype="<i2")
    captured[true_delay : true_delay + len(sig)] = ref
    # Light noise only — clean enough that the chirp peak is unambiguous.
    rng = np.random.default_rng(1)
    captured = (
        captured + (rng.standard_normal(len(captured)) * 50).astype("<i2")
    ).astype("<i2")

    ests = [
        estimate_playback_latency_ms(
            efference_pcm=ref, captured_pcm=captured[: len(ref)], sample_rate=sample_rate
        )
        for _ in range(8)
    ]
    ests = [e for e in ests if e is not None]
    assert len(ests) == 8
    assert max(ests) - min(ests) <= 1.0  # stable
    assert abs(float(np.mean(ests)) - true_delay / sample_rate * 1000.0) <= 1.0


def test_synthesize_tone_and_chirp_are_bounded_unit_waveforms() -> None:
    import numpy as np

    from cora_infans.io.motor_plan import synthesize_chirp, synthesize_tone

    tone = synthesize_tone(freq_hz=1000.0, duration_ms=200, sample_rate=16000)
    chirp = synthesize_chirp(duration_ms=200, sample_rate=16000)
    for sig in (tone, chirp):
        assert sig.dtype == np.float32
        assert abs(float(np.max(np.abs(sig))) - 1.0) < 1e-3  # normalized to [-1, 1]
        assert len(sig) == int(0.2 * 16000)


def test_latency_estimator_rejects_room_tone_dominated_trials() -> None:
    # A 5%-strength playback buried in full-scale room noise is not legitimate
    # calibration evidence, even when a causal search window is available.
    import numpy as np

    from cora_infans.io.motor_plan import synthesize_chirp
    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    sample_rate = 16000
    true_delay = 53  # samples
    rng = np.random.default_rng(3)
    sig = synthesize_chirp(duration_ms=300, sample_rate=sample_rate)
    ref = (sig * 30000).astype("<i2")

    def _estimates(causal_window_ms):
        out = []
        for _ in range(12):
            n = len(sig) + true_delay
            # Weak played signal embedded in strong noise (room-tone dominated).
            noise = rng.standard_normal(n) * 30000.0
            cap = np.zeros(n, dtype="<i2")
            cap[true_delay : true_delay + len(sig)] = ref
            cap = (cap.astype(np.float64) * 0.05 + noise).astype("<i2")
            e = estimate_playback_latency_ms(
                efference_pcm=ref,
                captured_pcm=cap[: len(ref)],
                sample_rate=sample_rate,
                causal_window_ms=causal_window_ms,
            )
            if e is not None:
                out.append(e)
        return out

    free = _estimates(None)
    windowed = _estimates(8.0)
    assert free == []
    assert windowed == []


def test_real_playback_passes_volume_db_to_ffmpeg(monkeypatch) -> None:
    # Verify the volume parameter is forwarded into the ffmpeg command (without
    # actually invoking ffmpeg) so higher/lower gains are honored.
    import cora_infans.live_speak_smoke as sm

    captured: dict[str, object] = {}

    class _FakePopen:
        returncode = 0

        def wait(self) -> None:
            captured["waited"] = True

    def _fake_popen(cmd, **kw):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["kwargs"] = kw
        return _FakePopen()

    monkeypatch.setattr(sm.subprocess, "Popen", _fake_popen)
    sm.play_wav_to_device("/tmp/x.wav", device_index=2, volume_db=-6.0)
    assert captured["waited"]
    assert any("volume=-6.0dB" in str(c) for c in captured["cmd"])
    assert captured["kwargs"]["stdout"] is sm.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is sm.subprocess.DEVNULL


def test_real_playback_raises_when_ffmpeg_fails(monkeypatch) -> None:
    import pytest

    import cora_infans.live_speak_smoke as sm

    class _FailedPopen:
        returncode = 1

        def wait(self) -> None:
            return None

    monkeypatch.setattr(sm.subprocess, "Popen", lambda *args, **kwargs: _FailedPopen())

    with pytest.raises(RuntimeError, match="playback failed"):
        sm.play_wav_to_device("/tmp/x.wav", device_index=2)


def test_capture_to_wav_signals_ready_after_raw_pcm_frames(tmp_path, monkeypatch) -> None:
    import threading

    import cora_infans.live_speak_smoke as sm

    path = tmp_path / "capture.wav"
    raw_path = Path(f"{path}.capture.raw")
    path.write_bytes(b"stale" * 20)
    raw_path.write_bytes(b"stale")

    class _FakeCaptureProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self):
            self.returncode = 0

    def fake_popen(cmd, **kwargs):
        assert not path.exists()
        assert not raw_path.exists()
        raw_path.write_bytes(np.zeros(8, dtype="<i2").tobytes())
        return _FakeCaptureProcess()

    monkeypatch.setattr(sm.subprocess, "Popen", fake_popen)
    ready = threading.Event()
    timeline: dict[str, object] = {}

    result = sm.capture_to_wav(
        str(path),
        device=":0",
        sample_rate=48000,
        duration_ms=100,
        ready_event=ready,
        capture_timeline=timeline,
    )

    assert ready.is_set()
    assert result["completed"] is True
    assert path.exists()


def test_capture_to_wav_honors_startup_cancellation(tmp_path, monkeypatch) -> None:
    import threading

    import pytest

    import cora_infans.live_speak_smoke as sm

    path = tmp_path / "capture.wav"

    class FakeProcess:
        returncode = None
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(sm.subprocess, "Popen", lambda *args, **kwargs: process)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(RuntimeError, match="live capture .* was cancelled"):
        sm.capture_to_wav(
            str(path),
            device=":0",
            duration_ms=100,
            cancel_event=cancelled,
            startup_timeout_ms=20,
        )

    assert process.killed is True
    assert not Path(f"{path}.capture.raw").exists()


def test_latency_estimator_rejects_silent_capture() -> None:
    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    sample_rate = 16000
    phase = np.arange(1600, dtype=np.float64)
    reference = np.sin(phase * 0.1)

    assert (
        estimate_playback_latency_ms(
            efference_pcm=reference,
            captured_pcm=np.zeros_like(reference),
            sample_rate=sample_rate,
            causal_window_ms=4.0,
        )
        is None
    )


def test_latency_estimator_rejects_unrelated_non_silent_capture() -> None:
    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    rng = np.random.default_rng(7)
    reference = rng.normal(size=8000).astype(np.float32)
    unrelated = rng.normal(size=8000).astype(np.float32)

    latency = estimate_playback_latency_ms(
        efference_pcm=reference,
        captured_pcm=unrelated,
        sample_rate=16000,
        causal_window_ms=20.0,
    )

    assert latency is None


def test_latency_estimator_rejects_peak_at_search_boundary() -> None:
    from cora_infans.live_speak_smoke import estimate_playback_latency_ms

    sample_rate = 16000
    boundary_samples = 320  # 20 ms
    rng = np.random.default_rng(11)
    reference = rng.normal(size=4000).astype(np.float32)
    captured = np.zeros_like(reference)
    captured[boundary_samples:] = reference[:-boundary_samples]

    assert (
        estimate_playback_latency_ms(
            efference_pcm=reference,
            captured_pcm=captured,
            sample_rate=sample_rate,
            causal_window_ms=20.0,
        )
        is None
    )


def test_read_wav_mono_float32_downmixes_real_stereo_frames(tmp_path) -> None:
    from cora_infans.live_speak_smoke import _read_wav_mono_float32

    path = tmp_path / "stereo.wav"
    stereo = np.asarray([[32767, -32768], [16384, -16384]], dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(stereo.tobytes())

    pcm, sample_rate = _read_wav_mono_float32(str(path))

    assert sample_rate == 48000
    assert pcm.shape == (2,)
    assert np.allclose(pcm, [-1.0 / 65536.0, 0.0], atol=1e-7)


def test_capture_during_playback_overlaps_and_joins_capture(tmp_path) -> None:
    import threading

    from cora_infans.live_speak_smoke import _capture_during_playback

    capture_active = threading.Event()
    playback_finished = threading.Event()
    order: list[str] = []

    def capture(
        path: str, *, sample_rate: int, duration_ms: int, ready_event
    ) -> dict[str, object]:
        order.append("capture_start")
        capture_active.set()
        ready_event.set()
        assert playback_finished.wait(timeout=1.0)
        order.append("capture_end")
        return {"completed": True}

    def playback(path: str) -> dict[str, object]:
        assert capture_active.wait(timeout=1.0)
        order.append("playback")
        playback_finished.set()
        return {"completed": True}

    result = _capture_during_playback(
        capture_path=str(tmp_path / "captured.wav"),
        playback_path=str(tmp_path / "efference.wav"),
        sample_rate=48000,
        capture_duration_ms=400,
        capture_fn=capture,
        playback_fn=playback,
        capture_preroll_ms=0,
    )

    assert order == ["capture_start", "playback", "capture_end"]
    assert result["playback"]["completed"] is True
    assert result["capture"]["completed"] is True
    assert result["playback_onset_ms"] >= result["capture_start_ms"]


def test_capture_coordinator_waits_for_explicit_readiness(tmp_path) -> None:
    import threading
    import time

    from cora_infans.live_speak_smoke import _capture_during_playback

    release = threading.Event()
    order: list[str] = []
    capture_started = time.monotonic()

    def capture(path, *, sample_rate, duration_ms, ready_event, capture_timeline):
        capture_timeline["frame_position_fn"] = lambda: int(
            (time.monotonic() - capture_started) * sample_rate
        )
        capture_timeline["position_source"] = "synthetic_frame_count"
        time.sleep(0.05)
        order.append("capture_ready")
        ready_event.set()
        assert release.wait(timeout=1.0)
        return {"completed": True}

    def playback(path):
        order.append("playback")
        release.set()
        return {"completed": True}

    result = _capture_during_playback(
        capture_path=str(tmp_path / "capture.wav"),
        playback_path=str(tmp_path / "playback.wav"),
        sample_rate=48000,
        capture_duration_ms=400,
        capture_fn=capture,
        playback_fn=playback,
        capture_preroll_ms=100,
    )

    assert order == ["capture_ready", "playback"]
    assert result["overlap_verified"] is True
    assert result["timestamp_source"] == "synthetic_frame_count"
    assert 140 <= result["playback_offset_from_capture_ms"] <= 180


def test_capture_coordinator_readiness_timeout_returns_promptly(tmp_path) -> None:
    import threading
    import time

    import pytest

    from cora_infans.live_speak_smoke import _capture_during_playback

    cancelled = threading.Event()

    def capture(path, *, sample_rate, duration_ms, ready_event, cancel_event):
        assert cancel_event.wait(timeout=1.0)
        cancelled.set()
        return {"completed": True}

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="capture did not become ready before timeout"):
        _capture_during_playback(
            capture_path=str(tmp_path / "capture.wav"),
            playback_path=str(tmp_path / "playback.wav"),
            sample_rate=48000,
            capture_duration_ms=400,
            capture_fn=capture,
            playback_fn=lambda path: {"completed": True},
            capture_preroll_ms=0,
            capture_ready_timeout_ms=20,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 0.1
    assert cancelled.wait(timeout=0.1)


def test_capture_default_ready_timeout_allows_slow_device_startup(tmp_path) -> None:
    """AVFoundation can take just over a second to publish its first frame."""
    import time

    from cora_infans.live_speak_smoke import _capture_during_playback

    def slow_capture(path, *, ready_event, **kwargs):
        time.sleep(1.05)
        ready_event.set()
        time.sleep(0.05)
        return {"completed": True}

    result = _capture_during_playback(
        capture_path=str(tmp_path / "capture.wav"),
        playback_path=str(tmp_path / "playback.wav"),
        sample_rate=48000,
        capture_duration_ms=1500,
        capture_fn=slow_capture,
        playback_fn=lambda path: {"completed": True},
        capture_preroll_ms=0,
    )

    assert result["overlap_verified"] is True


def test_capture_coordinator_rejects_capture_ending_before_playback(tmp_path) -> None:
    import time

    import pytest

    from cora_infans.live_speak_smoke import _capture_during_playback

    def capture(path, *, sample_rate, duration_ms, ready_event):
        ready_event.set()
        time.sleep(0.01)
        return {"completed": True}

    def playback(path):
        time.sleep(0.04)
        return {"completed": True}

    with pytest.raises(RuntimeError, match="capture ended before playback finished"):
        _capture_during_playback(
            capture_path=str(tmp_path / "capture.wav"),
            playback_path=str(tmp_path / "playback.wav"),
            sample_rate=48000,
            capture_duration_ms=400,
            capture_fn=capture,
            playback_fn=playback,
            capture_preroll_ms=0,
        )


def test_save_latency_calibration_writes_artifact(tmp_path) -> None:
    # Pure unit: a stable measurement is persisted as an honest, auditable artifact
    # carrying the measurement conditions (no biological claim).
    import json as _json

    from cora_infans.live_speak_smoke import save_latency_calibration

    path = str(tmp_path / "calibration" / "latency.json")
    returned = save_latency_calibration(
        path,
        latency_ms=0.844,
        stimulus_key="c200-4000",
        volume_db=24.0,
        causal_window_ms=4.0,
        sample_rate=48000,
        trials=6,
        std_ms=0.83,
        synchronized_capture=True,
    )
    assert returned == path
    cal = _json.loads(open(path).read())
    assert cal["kind"] == "playback_capture_loopback_latency"
    assert cal["latency_ms"] == 0.844
    assert cal["std_ms"] == 0.83
    assert cal["stimulus_key"] == "c200-4000"
    assert cal["volume_db"] == 24.0
    assert cal["causal_window_ms"] == 4.0
    assert cal["trials"] == 6
    assert cal["contamination_policy"] == "tag_only"
    assert "recorded_at" in cal
