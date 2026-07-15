"""Contracts for the integrated progress hero demo."""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time
import wave

import numpy as np
import pytest

from cora_infans.hero_demo import run_hero_demo
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.product_session import score_product_session_directory


def _write_product_session_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _make_product_session_fixture(root: Path) -> Path:
    session = root / "product-session"
    session.mkdir()
    sample_rate = 16000
    time = np.arange(int(sample_rate * 0.8)) / sample_rate
    samples = (0.25 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    for filename in (
        "source_human.wav",
        "target_captured.wav",
        "agent_emitted.wav",
        "response_captured.wav",
        "self_prediction.wav",
    ):
        _write_product_session_wav(session / filename, samples, sample_rate)
    score_product_session_directory(
        session,
        evidence_class="synthetic_fixture",
        association_frozen_before_evaluation=True,
    )
    return session


def test_replay_hero_exposes_learning_recall_and_workspace(tmp_path) -> None:
    report = run_hero_demo(
        output_dir=tmp_path,
        source="synthetic",
        mode="replay",
        vowel="a",
    )

    assert report["demo_passed"] is True
    assert report["evidence_class"] == "presentation_fixture"
    assert report["real_hardware"] is False
    assert report["live_playback"] is False
    assert len(report["prototype_map"]) == 96
    attempts = report["source"]["learning"]["attempts"]
    assert len(attempts) == 12
    assert len({row["prototype_id"] for row in attempts[-3:]}) == 1
    assert [row["phase"] for row in attempts[-3:]] == [
        "learned_recall",
        "learned_recall",
        "learned_recall",
    ]
    outcomes = report["workspace"]["outcomes"]
    assert outcomes["self_probe"]["decision"] == "suppress"
    assert outcomes["world_only"]["decision"] == "preserve"
    assert outcomes["overlap"]["decision"] == "preserve"
    assert outcomes["wrong_efference"]["decision"] == "preserve"
    assert report["retained_latency"]["status"] == (
        "retained_not_measured_this_session"
    )
    assert (tmp_path / "hero-demo.html").is_file()
    assert (tmp_path / "hero-demo.json").is_file()
    dashboard = (tmp_path / "hero-demo.html").read_text()
    assert "From hearing" in dashboard
    assert "Hear the journey" in dashboard
    assert "Honest boundary" in dashboard


def test_product_session_hero_presents_verified_replay_evidence(tmp_path) -> None:
    session = _make_product_session_fixture(tmp_path)

    report = run_hero_demo(
        output_dir=tmp_path / "presentation",
        source="product-session",
        session_dir=session,
        mode="replay",
    )

    assert report["mode"] == "product_session_hero_replay"
    assert report["evidence_class"] == "synthetic_fixture"
    assert report["real_hardware"] is False
    assert report["live_playback"] is False
    assert report["product_session"]["manifest_verified"] is True
    assert set(report["product_session"]["scores"]) == {
        "imitation_fidelity",
        "self_model_fidelity",
    }
    dashboard = (tmp_path / "presentation" / "hero-demo.html").read_text()
    assert "Product session evidence" in dashboard
    assert "REPLAY ONLY · NO NEW AUDIO CAPTURE" in dashboard
    assert "Imitation fidelity" in dashboard
    assert "Self-model fidelity" in dashboard
    assert "RMS-envelope distance" in dashboard
    assert "session-manifest.json" in dashboard


def test_product_session_hero_rejects_tampered_manifest_before_copying_audio(tmp_path) -> None:
    session = _make_product_session_fixture(tmp_path)
    (session / "session-manifest.json").write_text(
        '{"report_path":"report.json","report_sha256":"' + "0" * 64 + '"}'
    )
    presentation = tmp_path / "presentation"

    with pytest.raises(ValueError, match="manifest does not verify report.json"):
        run_hero_demo(
            output_dir=presentation,
            source="product-session",
            session_dir=session,
            mode="replay",
        )

    assert not list(presentation.glob("*.wav"))


def test_replay_hero_is_deterministically_reproducible(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = run_hero_demo(output_dir=first, source="synthetic", mode="replay")
    second_report = run_hero_demo(output_dir=second, source="synthetic", mode="replay")

    assert first_report == second_report
    for relative in (
        "teacher.wav",
        "learned.wav",
        "tracks/self-exact.wav",
        "captures/overlap.wav",
        "hero-demo.html",
        "hero-demo.json",
    ):
        first_bytes = (first / relative).read_bytes()
        second_bytes = (second / relative).read_bytes()
        assert first_bytes == second_bytes
        assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(
            second_bytes
        ).hexdigest()


def test_live_hero_refuses_before_io_without_explicit_authorization(tmp_path) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="authorize-live-io"):
        run_hero_demo(
            output_dir=tmp_path,
            mode="live",
            authorized_live_io=False,
            preflight_report={"preflight_passed": True},
            self_playback=lambda path: calls.append(path) or {"started": True},
            world_playback=lambda path: calls.append(path) or {"started": True},
            capture_fn=lambda path, **kwargs: calls.append(path) or {},
            allow_test_fixture=True,
        )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_injected_live_fixture_exercises_exact_routing_without_real_claim(
    tmp_path,
) -> None:
    playback_started = threading.Event()
    playback_calls: list[str] = []

    def playback(path: str) -> dict[str, object]:
        playback_calls.append(Path(path).name)
        playback_started.set()
        return {"backend": "injected", "started": True, "completed": True}

    def capture(
        path: str,
        *,
        sample_rate: int,
        duration_ms: int,
        ready_event,
        capture_timeline,
        **_kwargs,
    ) -> dict[str, object]:
        playback_started.clear()
        offset = sample_rate // 10
        capture_timeline["frame_position_fn"] = lambda: offset
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_started.wait(timeout=2.0)
        # The injected playback callback is intentionally instantaneous. Keep this
        # fake capture active long enough to model a real capture interval, so the
        # production invariant (capture outlasts playback) is tested deterministically.
        time.sleep(0.15)

        condition = Path(path).stem
        tracks = tmp_path / "tracks"
        if condition in {"self_calibration", "self_probe"}:
            names = ("self-exact.wav",)
        elif condition == "wrong_reference":
            names = ("wrong-reference.wav",)
        elif condition == "world_only":
            names = ("world-only.wav",)
        else:
            names = ("self-exact.wav", "world-overlap.wav")
        sources = [np.mean(read_pcm_wav(tracks / name)[0], axis=1) for name in names]
        mixed = np.sum(np.stack(sources), axis=0)
        total = int(sample_rate * duration_ms / 1000)
        rendered = np.zeros(total, dtype=np.float32)
        available = min(len(mixed), total - offset)
        rendered[offset : offset + available] = 0.68 * mixed[:available]
        stereo = np.repeat(rendered[:, None], 2, axis=1)
        pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(path, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return {
            "backend": "injected_capture",
            "frames": total,
            "completed": True,
        }

    report = run_hero_demo(
        output_dir=tmp_path,
        source="synthetic",
        mode="live",
        vowel="a",
        authorized_live_io=True,
        preflight_report={"preflight_passed": True},
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
        allow_test_fixture=True,
    )

    assert report["demo_passed"] is True
    assert report["evidence_class"] == "synthetic_injected_live_fixture"
    assert report["real_hardware"] is False
    assert report["live_playback"] is False
    assert report["workspace"]["panel_passed"] is True
    assert "self-exact.wav" in playback_calls
    assert "world-overlap.wav" in playback_calls
    assert all(
        capture["real_hardware"] is False
        for capture in report["workspace"]["captures"].values()
    )
