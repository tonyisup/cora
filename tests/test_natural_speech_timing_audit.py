from __future__ import annotations

import json

from cora_infans.natural_speech_timing_audit import (
    EXPECTED_STIMULUS_SHA256,
    audit_diverse_natural_speech,
    audit_natural_speech_diagnostic,
)


def _report(*, clipped_playback: int = 0, clipped_trailing: int = 0):
    frames = []
    for index in range(400):
        playback = index < 270
        clipped = (
            clipped_playback
            if index == 20 and playback
            else clipped_trailing
            if index == 300 and not playback
            else 0
        )
        frames.append(
            {
                "stimulus_playback_active": playback,
                "clipped_sample_count": clipped,
                "capture_peak_absolute": 1.0 if clipped else 0.5,
            }
        )
    total_clipped = clipped_playback + clipped_trailing
    return {
        "frame_count": 400,
        "salient_frame_count": 200,
        "frames": frames,
        "stimulus": {
            "source_wav_sha256": EXPECTED_STIMULUS_SHA256,
            "volume_db": -18.0,
        },
        "capture_quality": {
            "clipped_sample_count": total_clipped,
            "peak_absolute": 1.0 if total_clipped else 0.5,
        },
        "runtime_performance": {
            "real_capture": True,
            "capture_backend": "sounddevice_portaudio_coreaudio_callback",
            "workload_latency_ms": {
                "salient": {"adc_frame_end_to_workspace_decision": {"p95": 10.0}},
                "quiet": {"adc_frame_end_to_workspace_decision": {"p95": 10.0}},
            },
            "device_timestamped_deadline_miss_count": 0,
            "input_overflow_count": 0,
            "steady_state_real_time_factor": 1.0,
        },
        "runtime_resources": {
            "average_process_cpu_percent": 4.0,
            "peak_resident_memory_megabytes": 65.0,
        },
        "capture_callback_status_count": 0,
        "capture_callback_timestamp_discontinuity_count": 0,
        "capture_stream_fallback_reason": None,
    }


def test_natural_speech_diagnostic_reconciles_and_partitions_clipping(tmp_path) -> None:
    paths = []
    for index, report in enumerate(
        (_report(), _report(clipped_playback=3), _report(clipped_trailing=2))
    ):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)

    result = audit_natural_speech_diagnostic(paths)

    assert result["runs"][1]["playback_interval"]["clipped_sample_count"] == 3
    assert result["runs"][2]["trailing_quiet_interval"]["clipped_sample_count"] == 2
    assert result["criteria"]["frame_clipping_sum_equals_aggregate_each"] is True
    assert result["criteria"]["real_capture_each"] is True
    assert result["criteria"]["callback_backend_each"] is True
    assert result["criteria"]["clipped_sample_count_each_max_0"] is False
    assert result["promotion_allowed"] is False


def test_diverse_natural_speech_requires_every_frozen_stimulus_to_pass(tmp_path) -> None:
    reports = []
    stimuli = []
    for index in range(6):
        digest = f"{index + 1:064x}"
        report = _report(clipped_playback=1 if index == 4 else 0)
        report["frame_count"] = 800
        report["frames"] = report["frames"] * 2
        report["stimulus"]["source_wav_sha256"] = digest
        path = tmp_path / f"diverse-{index}.json"
        path.write_text(json.dumps(report))
        reports.append(path)
        stimuli.append(
            {
                "filename": f"SITE{index}.wav",
                "sha256": digest,
                "fieldsite": f"S{index}",
                "speaker": f"S{index}01",
                "condition": (
                    "infant_directed_speech"
                    if index % 2
                    else "adult_directed_speech"
                ),
            }
        )
    protocol = {
        "experiment": "stage6_diverse_natural_speech_timing_and_integrity",
        "stimuli": stimuli,
        "fixed_runtime": {
            "target_frames_each": 800,
            "software_playback_gain_db": -18.0,
        },
    }

    result = audit_diverse_natural_speech(reports, protocol)

    assert result["stimulus_count"] == 6
    assert result["fieldsite_count"] == 6
    assert result["condition_counts"] == {
        "adult_directed_speech": 3,
        "infant_directed_speech": 3,
    }
    assert result["criteria"]["clipped_sample_count_each_max_0"] is False
    assert result["failed_stimuli"] == ["SITE4.wav"]
    assert result["promotion_allowed"] is False
