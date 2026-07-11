"""Explicit-boundary latency measurements for live and injected runtimes."""

from __future__ import annotations

import statistics

import numpy as np


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": max(values),
    }


class RuntimeLatencyTracker:
    """Accumulates capture-call and workspace-decision latency per frame."""

    def __init__(
        self,
        *,
        deadline_ms: float,
        deadline_tolerance_ms: float | None = None,
    ) -> None:
        if deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")
        self.deadline_ms = float(deadline_ms)
        self.deadline_tolerance_ms = (
            float(deadline_tolerance_ms)
            if deadline_tolerance_ms is not None
            else max(5.0, self.deadline_ms * 0.05)
        )
        if self.deadline_tolerance_ms < 0.0:
            raise ValueError("deadline_tolerance_ms must be non-negative")
        self.capture_ms: list[float] = []
        self.pipeline_ms: list[float] = []
        self.total_ms: list[float] = []
        self.adc_onset_to_decision_ms: list[float] = []
        self.adc_end_to_decision_ms: list[float] = []

    def record(
        self,
        *,
        capture_ms: float,
        pipeline_ms: float,
        total_ms: float,
        adc_onset_to_decision_ms: float | None = None,
        adc_end_to_decision_ms: float | None = None,
    ) -> None:
        if min(capture_ms, pipeline_ms, total_ms) < 0.0:
            raise ValueError("latencies must be non-negative")
        self.capture_ms.append(float(capture_ms))
        self.pipeline_ms.append(float(pipeline_ms))
        self.total_ms.append(float(total_ms))
        if (adc_onset_to_decision_ms is None) != (adc_end_to_decision_ms is None):
            raise ValueError("ADC onset and end latencies must be recorded together")
        if adc_onset_to_decision_ms is not None and adc_end_to_decision_ms is not None:
            if min(adc_onset_to_decision_ms, adc_end_to_decision_ms) < 0.0:
                raise ValueError("ADC latencies must be non-negative")
            self.adc_onset_to_decision_ms.append(float(adc_onset_to_decision_ms))
            self.adc_end_to_decision_ms.append(float(adc_end_to_decision_ms))

    def summary(
        self,
        *,
        capture_backend: str,
        real_capture: bool,
        backend_reported_input_latency_ms: float | None = None,
        input_overflow_count: int = 0,
    ) -> dict[str, object]:
        if input_overflow_count < 0:
            raise ValueError("input_overflow_count must be non-negative")
        strict_misses = sum(value > self.deadline_ms for value in self.total_ms)
        operational_limit = self.deadline_ms + self.deadline_tolerance_ms
        misses = sum(value > operational_limit for value in self.total_ms)
        device_strict_misses = sum(
            value > self.deadline_ms for value in self.adc_end_to_decision_ms
        )
        device_misses = sum(
            value > operational_limit for value in self.adc_end_to_decision_ms
        )
        total_measured_ms = sum(self.total_ms)
        represented_ms = len(self.total_ms) * self.deadline_ms
        steady_state = self.total_ms[1:]
        steady_represented_ms = len(steady_state) * self.deadline_ms
        report: dict[str, object] = {
            "mode": "live_capture_to_workspace_latency",
            "measurement_boundary": "capture_frame_call_start_to_workspace_decision",
            "timestamp_quality": (
                "portaudio_inputBufferAdcTime_to_stream_time"
                if self.adc_end_to_decision_ms
                else "host_monotonic_call_boundary_not_audio_device_onset"
            ),
            "capture_backend": capture_backend,
            "real_capture": real_capture,
            "deadline_ms": self.deadline_ms,
            "deadline_tolerance_ms": self.deadline_tolerance_ms,
            "frame_count": len(self.total_ms),
            "strict_deadline_miss_count": strict_misses,
            "strict_deadline_miss_rate": (
                strict_misses / len(self.total_ms) if self.total_ms else 0.0
            ),
            "deadline_miss_count": misses,
            "deadline_miss_rate": misses / len(self.total_ms) if self.total_ms else 0.0,
            "input_overflow_count": input_overflow_count,
            "input_overflow_rate": (
                input_overflow_count / len(self.total_ms) if self.total_ms else 0.0
            ),
            "device_timestamped_strict_deadline_miss_count": (
                device_strict_misses if self.adc_end_to_decision_ms else None
            ),
            "device_timestamped_strict_deadline_miss_rate": (
                device_strict_misses / len(self.adc_end_to_decision_ms)
                if self.adc_end_to_decision_ms
                else None
            ),
            "device_timestamped_deadline_miss_count": (
                device_misses if self.adc_end_to_decision_ms else None
            ),
            "device_timestamped_deadline_miss_rate": (
                device_misses / len(self.adc_end_to_decision_ms)
                if self.adc_end_to_decision_ms
                else None
            ),
            "dropped_frame_proxy": (
                "PortAudio input-overflow flag per requested analysis frame"
            ),
            "requested_frame_rate_hz": 1000.0 / self.deadline_ms,
            "measured_path_real_time_factor": (
                total_measured_ms / represented_ms if represented_ms else 0.0
            ),
            "cold_start_capture_to_workspace_ms": (
                self.total_ms[0] if self.total_ms else None
            ),
            "steady_state_capture_to_workspace_latency_ms": _distribution(steady_state),
            "steady_state_real_time_factor": (
                sum(steady_state) / steady_represented_ms
                if steady_represented_ms
                else 0.0
            ),
            "capture_call_latency_ms": _distribution(self.capture_ms),
            "pipeline_decision_latency_ms": _distribution(self.pipeline_ms),
            "capture_to_workspace_latency_ms": _distribution(self.total_ms),
            "device_timestamped_frame_latency_ms": (
                {
                    "adc_frame_onset_to_workspace_decision": _distribution(
                        self.adc_onset_to_decision_ms
                    ),
                    "adc_frame_end_to_workspace_decision": _distribution(
                        self.adc_end_to_decision_ms
                    ),
                    "timestamp_source": (
                        "PortAudioCallbackTimeInfo.inputBufferAdcTime and "
                        "Pa_GetStreamTime"
                    ),
                }
                if self.adc_end_to_decision_ms
                else None
            ),
            "excluded": [
                *(
                    ["raw_coreaudio_AudioTimeStamp"]
                    if self.adc_end_to_decision_ms
                    else ["hardware_adc_sample_onset", "coreaudio_backend_timestamp"]
                ),
                "xmos_internal_dsp_latency",
                "attention_usb_steer_after_workspace_decision",
                "energy_measurement",
            ],
        }
        report["backend_reported_input_latency_ms"] = backend_reported_input_latency_ms
        report["estimated_adc_to_workspace_latency_ms"] = (
            _distribution(
                [
                    value + backend_reported_input_latency_ms
                    for value in self.total_ms
                ]
            )
            if backend_reported_input_latency_ms is not None
            else None
        )
        return report
