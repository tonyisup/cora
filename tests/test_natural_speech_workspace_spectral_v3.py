from pathlib import Path
from threading import Event
import json
import time

import numpy as np
import pytest

from cora_infans.io.motor_plan import write_wav
from cora_infans.io.wav_frames import read_pcm_wav
from cora_infans.natural_speech_workspace_spectral_v3 import (
    FROZEN_THRESHOLD,
    audit_spent_v2_design,
    build_spectral_protocol,
    run_spectral_live_acquisition,
)


SPENT_ROOT = Path("/private/tmp/cora-ids-spent-200")
V2_ARTIFACTS = Path(
    "docs/evidence/2026-07-11-natural-speech-workspace-template-v2-real-artifacts"
)
V2_RESULT = Path(
    "docs/evidence/2026-07-11-natural-speech-workspace-template-v2-real-result.json"
)


def test_spectral_v3_protocol_is_final_bounded_fresh_gate() -> None:
    protocol = build_spectral_protocol()
    assert protocol["trial_count"] == 12
    assert protocol["scored_trial_count"] == 9
    assert protocol["maximum_total_playback_seconds"] == pytest.approx(11.04)
    assert protocol["frozen_threshold"] == FROZEN_THRESHOLD
    assert protocol["template_memory"]["scalars_per_slot"] == 12


def test_retained_fresh_spectral_v3_result_passes_narrow_gate() -> None:
    result = json.loads(
        Path(
            "docs/evidence/2026-07-11-natural-speech-workspace-spectral-v3-real-result.json"
        ).read_text()
    )
    selected = result["reports"]["3"]
    assert result["verdict"] == "real_io_acceptance_passed"
    assert result["promotion_allowed"] is True
    assert all(result["criteria"].values())
    assert result["capture_quality"]["total_clipped_samples"] == 0
    assert selected["scheduled_identity_plus_spectral_template"]["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5


@pytest.mark.skipif(
    not V2_ARTIFACTS.exists() or not SPENT_ROOT.exists(),
    reason="retained v2 evidence unavailable",
)
def test_spent_v2_spectral_design_gate_passes() -> None:
    report = audit_spent_v2_design(
        artifacts=V2_ARTIFACTS,
        result=V2_RESULT,
        corpus_root=SPENT_ROOT,
    )
    assert report["verdict"] == "spent_design_gate_passed"
    assert all(report["criteria"].values())
    assert max(report["self_probe_distances"]) < FROZEN_THRESHOLD
    assert min(report["overlap_distances"]) >= FROZEN_THRESHOLD
    assert report["promotion_allowed"] is False


@pytest.mark.skipif(not SPENT_ROOT.exists(), reason="external spent corpus unavailable")
def test_spectral_v3_injected_acquisition_passes_without_promotion(tmp_path) -> None:
    playback_seen = Event()

    def playback(_path):
        playback_seen.set()
        return {"backend": "injected_playback", "started": True}

    def capture(path, *, sample_rate, duration_ms, ready_event, capture_timeline, **_kwargs):
        playback_seen.clear()
        capture_timeline["frame_position_fn"] = lambda: sample_rate // 10
        capture_timeline["position_source"] = "synthetic_frame_count"
        ready_event.set()
        assert playback_seen.wait(timeout=1.0)
        time.sleep(0.003)
        trial_id = Path(path).name.removesuffix("_capture.wav")
        self_pcm, _ = read_pcm_wav(tmp_path / f"{trial_id}_self.wav")
        world_pcm, _ = read_pcm_wav(tmp_path / f"{trial_id}_world.wav")
        self_impulse = np.zeros(128)
        self_impulse[0], self_impulse[61] = 0.35, 0.07
        world_impulse = np.zeros(128)
        world_impulse[0], world_impulse[79] = 0.25, 0.04
        observed = np.convolve(np.mean(self_pcm, axis=1), self_impulse)
        observed += np.convolve(np.mean(world_pcm, axis=1), world_impulse)
        write_wav(path, np.concatenate((np.zeros(sample_rate // 10), observed, np.zeros(sample_rate // 10))), sample_rate)
        return {"backend": "injected_capture", "sample_rate": sample_rate, "duration_ms": duration_ms, "completed": True}

    report = run_spectral_live_acquisition(
        root=SPENT_ROOT,
        out_dir=tmp_path,
        self_playback=playback,
        world_playback=playback,
        capture_fn=capture,
    )
    assert report["verdict"] == "injected_acquisition_complete"
    assert report["completed_trial_count"] == 12
    assert report["reports"]["3"]["scheduled_identity_plus_spectral_template"]["metrics"]["f1"] == 1.0
    assert report["promotion_allowed"] is False
