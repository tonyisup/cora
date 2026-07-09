"""Fake-I/O SPEAK smoke harness before any real playback implementation."""

import wave
from pathlib import Path

import numpy as np

from cora_infans.live_speak_smoke import run_live_speak_smoke_trial


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


def test_real_io_runs_with_promoted_gate_using_injected_fakes(tmp_path) -> None:
    # With a promoted gate, the helper performs the wired playback/capture (here
    # injected fakes) and reports real_hardware=true with a finite latency.
    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trial

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    def _play(path: str) -> dict[str, object]:
        return {"backend": "fake", "device_index": 2, "completed": True}

    def _capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
        import wave

        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        return {"backend": "fake", "device": ":0", "frames": samples, "completed": True}

    report = run_real_io_speak_trial(
        vowel="a",
        out_dir=str(tmp_path),
        playback_fn=_play,
        capture_fn=_capture,
        readiness_gate_fn=_gate,
        sample_rate=16000,
    )

    assert report["verdict"] == "real_io_complete"
    assert report["real_hardware"] is True
    assert report["live_playback"] is True
    assert report["live_playback_allowed"] is False
    assert report["artifacts"]["captured_wav"] is not None
    assert report["latency_aligned_ms"] is not None
    assert isinstance(report["latency_aligned_ms"], (int, float))
    # Frame tags present and ordered.
    states = [t["speak_state"] for t in report["frame_tags"]]
    assert states[0] == "READY_TO_SPEAK"
    assert "SPEAKING" in states


def test_real_io_trials_aggregate_latency_stats_with_stable_estimator(tmp_path) -> None:
    # Multi-trial harness: runs N real trials and reports latency mean/std/min/max.
    # With a deterministic injected estimator the spread is zero and the harness
    # flags latency_stable=True (the precondition for any later subtraction).
    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trials

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    def _fake_play(path: str) -> dict[str, object]:
        return {"backend": "fake", "completed": True}

    def _fake_capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
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
    )

    assert report["verdict"] == "trials_complete"
    assert report["trials_run"] == 4
    assert report["trials_ok"] == 4
    assert report["latency_stats"]["mean"] == 12.0
    assert report["latency_stats"]["std"] == 0.0
    assert report["latency_stats"]["min"] == 12.0
    assert report["latency_stats"]["max"] == 12.0
    assert report["latency_stable"] is True


def test_real_io_trials_flag_unstable_when_spread_exceeds_threshold(tmp_path) -> None:
    # If the per-trial latency varies beyond the jitter threshold, the harness must
    # NOT claim stability — honest precondition for signal-level subtraction.
    import numpy as np

    from cora_infans.live_speak_smoke import run_real_io_speak_trials

    def _gate() -> dict[str, object]:
        return {"verdict": "promote_simulation_only", "live_playback_allowed": False}

    def _fake_play(path: str) -> dict[str, object]:
        return {"backend": "fake", "completed": True}

    def _fake_capture(path: str, *, sample_rate: int, duration_ms: int) -> dict[str, object]:
        samples = int(sample_rate * duration_ms / 1000)
        pcm = np.zeros(samples, dtype="<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
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
    )

    assert report["trials_ok"] == 4
    assert report["latency_stats"]["std"] > 2.0
    assert report["latency_stable"] is False
