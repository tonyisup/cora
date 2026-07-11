"""Audit per-frame clipping and timing for the natural-speech live gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_STIMULUS_SHA256 = (
    "c58a46a68782fb318cd885e6c670025104cc844bb3d8e4080b69b1f995c2c410"
)


def _summarize_run(
    path: Path,
    *,
    expected_stimulus_sha256: str = EXPECTED_STIMULUS_SHA256,
    expected_frame_count: int = 400,
    expected_volume_db: float = -18.0,
) -> dict[str, object]:
    raw = path.read_bytes()
    report = json.loads(raw)
    frames = report.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frame_count:
        raise ValueError(
            f"natural-speech diagnostic requires {expected_frame_count} frames: {path}"
        )
    stimulus = report.get("stimulus")
    if not isinstance(stimulus, dict):
        raise ValueError(f"natural-speech diagnostic is missing stimulus: {path}")
    if stimulus.get("source_wav_sha256") != expected_stimulus_sha256:
        raise ValueError(f"natural-speech stimulus hash mismatch: {path}")
    if float(stimulus.get("volume_db", 0.0)) != expected_volume_db:
        raise ValueError(f"natural-speech software gain drift: {path}")

    playback_frames = [
        frame for frame in frames if bool(frame.get("stimulus_playback_active"))
    ]
    trailing_frames = [
        frame for frame in frames if not bool(frame.get("stimulus_playback_active"))
    ]

    def clipping(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "frame_count": len(rows),
            "clipped_frame_count": sum(
                int(frame.get("clipped_sample_count", 0)) > 0 for frame in rows
            ),
            "clipped_sample_count": sum(
                int(frame.get("clipped_sample_count", 0)) for frame in rows
            ),
            "maximum_frame_peak_absolute": max(
                (float(frame.get("capture_peak_absolute", 0.0)) for frame in rows),
                default=0.0,
            ),
        }

    playback = clipping(playback_frames)
    trailing = clipping(trailing_frames)
    capture_quality = report["capture_quality"]
    frame_clipped_sum = int(playback["clipped_sample_count"]) + int(
        trailing["clipped_sample_count"]
    )
    aggregate_clipped = int(capture_quality["clipped_sample_count"])
    performance = report["runtime_performance"]
    workloads = performance["workload_latency_ms"]
    resources = report["runtime_resources"]
    return {
        "raw_report_sha256": hashlib.sha256(raw).hexdigest(),
        "real_capture": bool(performance["real_capture"]),
        "capture_backend": str(performance["capture_backend"]),
        "frame_count": int(report["frame_count"]),
        "salient_frame_count": int(report["salient_frame_count"]),
        "playback_interval": playback,
        "trailing_quiet_interval": trailing,
        "aggregate_clipped_sample_count": aggregate_clipped,
        "frame_clipped_sample_sum": frame_clipped_sum,
        "frame_clipping_sum_equals_aggregate": frame_clipped_sum
        == aggregate_clipped,
        "capture_peak_absolute": float(capture_quality["peak_absolute"]),
        "salient_adc_frame_end_to_workspace_p95_ms": float(
            workloads["salient"]["adc_frame_end_to_workspace_decision"]["p95"]
        ),
        "quiet_adc_frame_end_to_workspace_p95_ms": float(
            workloads["quiet"]["adc_frame_end_to_workspace_decision"]["p95"]
        ),
        "device_deadline_miss_count": int(
            performance["device_timestamped_deadline_miss_count"]
        ),
        "input_overflow_count": int(performance["input_overflow_count"]),
        "callback_status_count": int(report["capture_callback_status_count"]),
        "timestamp_discontinuity_count": int(
            report["capture_callback_timestamp_discontinuity_count"]
        ),
        "capture_stream_fallback_reason": report["capture_stream_fallback_reason"],
        "steady_state_real_time_factor": float(
            performance["steady_state_real_time_factor"]
        ),
        "average_process_cpu_percent": float(
            resources["average_process_cpu_percent"]
        ),
        "peak_resident_memory_megabytes": float(
            resources["peak_resident_memory_megabytes"]
        ),
    }


def audit_natural_speech_diagnostic(paths: list[str | Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("natural-speech diagnostic requires exactly three runs")
    runs = [_summarize_run(Path(path)) for path in paths]
    criteria = {
        "real_capture_each": all(run["real_capture"] is True for run in runs),
        "callback_backend_each": all(
            run["capture_backend"]
            == "sounddevice_portaudio_coreaudio_callback"
            for run in runs
        ),
        "frame_count_each_400": all(run["frame_count"] == 400 for run in runs),
        "salient_frame_count_each_min_20": all(
            int(run["salient_frame_count"]) >= 20 for run in runs
        ),
        "input_overflow_count_each_max_0": all(
            run["input_overflow_count"] == 0 for run in runs
        ),
        "callback_status_count_each_max_0": all(
            run["callback_status_count"] == 0 for run in runs
        ),
        "timestamp_discontinuity_count_each_max_0": all(
            run["timestamp_discontinuity_count"] == 0 for run in runs
        ),
        "capture_stream_without_fallback_each": all(
            run["capture_stream_fallback_reason"] is None for run in runs
        ),
        "clipped_sample_count_each_max_0": all(
            run["aggregate_clipped_sample_count"] == 0 for run in runs
        ),
        "frame_clipping_sum_equals_aggregate_each": all(
            run["frame_clipping_sum_equals_aggregate"] is True for run in runs
        ),
        "actual_playback_and_trailing_quiet_partitions_nonempty_each": all(
            int(run["playback_interval"]["frame_count"]) > 0
            and int(run["trailing_quiet_interval"]["frame_count"]) > 0
            for run in runs
        ),
        "salient_adc_frame_end_to_workspace_p95_ms_each_max_30": all(
            float(run["salient_adc_frame_end_to_workspace_p95_ms"]) <= 30.0
            for run in runs
        ),
        "quiet_adc_frame_end_to_workspace_p95_ms_each_max_30": all(
            float(run["quiet_adc_frame_end_to_workspace_p95_ms"]) <= 30.0
            for run in runs
        ),
        "device_deadline_miss_count_each_max_0": all(
            run["device_deadline_miss_count"] == 0 for run in runs
        ),
        "steady_state_real_time_factor_each_max_1_2": all(
            float(run["steady_state_real_time_factor"]) <= 1.2 for run in runs
        ),
        "average_process_cpu_percent_each_max_100": all(
            float(run["average_process_cpu_percent"]) <= 100.0 for run in runs
        ),
        "peak_resident_memory_megabytes_each_max_512": all(
            float(run["peak_resident_memory_megabytes"]) <= 512.0 for run in runs
        ),
    }
    passed = all(criteria.values())
    return {
        "mode": "stage6_natural_speech_hardware_volume_clipping_audit",
        "hardware_volume": "approximately_50_percent_mechanical_travel",
        "runs": runs,
        "criteria": criteria,
        "overall_criteria_passed": passed,
        "promotion_allowed": passed,
    }


def audit_diverse_natural_speech(
    paths: list[str | Path],
    preregistration: str | Path | dict[str, object],
) -> dict[str, object]:
    """Audit the frozen six-fieldsite, one-run-per-stimulus live panel."""

    if isinstance(preregistration, dict):
        protocol = preregistration
        protocol_sha256 = None
    else:
        protocol_path = Path(preregistration)
        raw_protocol = protocol_path.read_bytes()
        protocol = json.loads(raw_protocol)
        protocol_sha256 = hashlib.sha256(raw_protocol).hexdigest()
    if protocol.get("experiment") != "stage6_diverse_natural_speech_timing_and_integrity":
        raise ValueError("unexpected diverse natural-speech preregistration")
    stimuli = protocol.get("stimuli")
    runtime = protocol.get("fixed_runtime")
    if not isinstance(stimuli, list) or len(stimuli) != 6:
        raise ValueError("diverse natural-speech audit requires six frozen stimuli")
    if not isinstance(runtime, dict):
        raise ValueError("diverse natural-speech audit is missing fixed runtime")
    if len(paths) != len(stimuli):
        raise ValueError("diverse natural-speech audit requires one run per stimulus")
    frame_count = int(runtime["target_frames_each"])
    volume_db = float(runtime["software_playback_gain_db"])
    runs = []
    for path, stimulus in zip(paths, stimuli, strict=True):
        if not isinstance(stimulus, dict):
            raise ValueError("invalid diverse natural-speech stimulus record")
        summary = _summarize_run(
            Path(path),
            expected_stimulus_sha256=str(stimulus["sha256"]),
            expected_frame_count=frame_count,
            expected_volume_db=volume_db,
        )
        summary.update(
            {
                "filename": str(stimulus["filename"]),
                "fieldsite": str(stimulus["fieldsite"]),
                "speaker": str(stimulus["speaker"]),
                "condition": str(stimulus["condition"]),
                "stimulus_sha256": str(stimulus["sha256"]),
            }
        )
        runs.append(summary)

    criteria = {
        "real_capture_each": all(run["real_capture"] is True for run in runs),
        "callback_backend_each": all(
            run["capture_backend"] == "sounddevice_portaudio_coreaudio_callback"
            for run in runs
        ),
        "frame_count_each_800": all(run["frame_count"] == 800 for run in runs),
        "salient_frame_count_each_min_20": all(
            int(run["salient_frame_count"]) >= 20 for run in runs
        ),
        "input_overflow_count_each_max_0": all(
            run["input_overflow_count"] == 0 for run in runs
        ),
        "callback_status_count_each_max_0": all(
            run["callback_status_count"] == 0 for run in runs
        ),
        "timestamp_discontinuity_count_each_max_0": all(
            run["timestamp_discontinuity_count"] == 0 for run in runs
        ),
        "capture_stream_without_fallback_each": all(
            run["capture_stream_fallback_reason"] is None for run in runs
        ),
        "clipped_sample_count_each_max_0": all(
            run["aggregate_clipped_sample_count"] == 0 for run in runs
        ),
        "frame_clipping_sum_equals_aggregate_each": all(
            run["frame_clipping_sum_equals_aggregate"] is True for run in runs
        ),
        "actual_playback_and_trailing_quiet_partitions_nonempty_each": all(
            int(run["playback_interval"]["frame_count"]) > 0
            and int(run["trailing_quiet_interval"]["frame_count"]) > 0
            for run in runs
        ),
        "salient_adc_frame_end_to_workspace_p95_ms_each_max_30": all(
            float(run["salient_adc_frame_end_to_workspace_p95_ms"]) <= 30.0
            for run in runs
        ),
        "quiet_adc_frame_end_to_workspace_p95_ms_each_max_30": all(
            float(run["quiet_adc_frame_end_to_workspace_p95_ms"]) <= 30.0
            for run in runs
        ),
        "device_deadline_miss_count_each_max_0": all(
            run["device_deadline_miss_count"] == 0 for run in runs
        ),
        "steady_state_real_time_factor_each_max_1_2": all(
            float(run["steady_state_real_time_factor"]) <= 1.2 for run in runs
        ),
        "average_process_cpu_percent_each_max_100": all(
            float(run["average_process_cpu_percent"]) <= 100.0 for run in runs
        ),
        "peak_resident_memory_megabytes_each_max_512": all(
            float(run["peak_resident_memory_megabytes"]) <= 512.0 for run in runs
        ),
    }
    passed = all(criteria.values())
    return {
        "mode": "stage6_diverse_natural_speech_timing_and_integrity_audit",
        "preregistration_sha256": protocol_sha256,
        "hardware_volume": "approximately_50_percent_mechanical_travel",
        "stimulus_count": len(runs),
        "fieldsite_count": len({str(run["fieldsite"]) for run in runs}),
        "speaker_count": len({str(run["speaker"]) for run in runs}),
        "condition_counts": {
            condition: sum(run["condition"] == condition for run in runs)
            for condition in sorted({str(run["condition"]) for run in runs})
        },
        "runs": runs,
        "criteria": criteria,
        "failed_stimuli": [
            str(run["filename"])
            for run in runs
            if not (
                run["real_capture"] is True
                and run["capture_backend"]
                == "sounddevice_portaudio_coreaudio_callback"
                and run["frame_count"] == 800
                and int(run["salient_frame_count"]) >= 20
                and run["input_overflow_count"] == 0
                and run["callback_status_count"] == 0
                and run["timestamp_discontinuity_count"] == 0
                and run["capture_stream_fallback_reason"] is None
                and run["aggregate_clipped_sample_count"] == 0
                and run["frame_clipping_sum_equals_aggregate"] is True
                and int(run["playback_interval"]["frame_count"]) > 0
                and int(run["trailing_quiet_interval"]["frame_count"]) > 0
                and float(run["salient_adc_frame_end_to_workspace_p95_ms"])
                <= 30.0
                and float(run["quiet_adc_frame_end_to_workspace_p95_ms"])
                <= 30.0
                and run["device_deadline_miss_count"] == 0
                and float(run["steady_state_real_time_factor"]) <= 1.2
                and float(run["average_process_cpu_percent"]) <= 100.0
                and float(run["peak_resident_memory_megabytes"]) <= 512.0
            )
        ],
        "overall_criteria_passed": passed,
        "promotion_allowed": passed,
    }
