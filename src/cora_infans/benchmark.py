"""Reproducible microbenchmarks for the bounded event-processing core."""

from __future__ import annotations

import os
import platform
import statistics
import time
import tracemalloc
from collections import deque

import numpy as np

from .config import CoraInfansConfig
from .pipeline import AudioFrame, CoraInfansPipeline
from .predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy
from .resource_metrics import ProcessResourceTracker


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class ConventionalFrameBaseline:
    """Fixed-threshold, nearest-prototype DSP baseline with bounded memory."""

    def __init__(self) -> None:
        self.events: deque[tuple[int, int]] = deque(maxlen=10)
        self.prototypes_hz = (300.0, 500.0, 700.0, 900.0)

    def process_frame(self, frame: AudioFrame) -> tuple[int, int] | None:
        energies = frame.channel_energies
        peak = max(energies)
        if peak - sum(energies) / len(energies) < 0.02:
            return None
        identity = min(
            range(len(self.prototypes_hz)),
            key=lambda index: abs(
                self.prototypes_hz[index] - frame.dominant_frequency_hz
            ),
        )
        event = (energies.index(peak), identity)
        self.events.append(event)
        return event


def _measure_processor(
    frames: list[AudioFrame], process_frame: object
) -> dict[str, object]:
    durations_ms: list[float] = []
    resources = ProcessResourceTracker()
    tracemalloc.start()
    for frame in frames:
        started = time.perf_counter_ns()
        process_frame(frame)  # type: ignore[operator]
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    resource_report = resources.summary()
    return {
        "latency_ms": {
            "mean": statistics.fmean(durations_ms),
            "p50": _percentile(durations_ms, 50.0),
            "p95": _percentile(durations_ms, 95.0),
            "p99": _percentile(durations_ms, 99.0),
            "max": max(durations_ms),
        },
        "wall_seconds": resource_report["wall_seconds"],
        "cpu_seconds": resource_report["cpu_seconds"],
        "average_process_cpu_percent": resource_report[
            "average_process_cpu_percent"
        ],
        "peak_resident_memory_bytes": resource_report[
            "peak_resident_memory_bytes"
        ],
        "peak_resident_memory_megabytes": resource_report[
            "peak_resident_memory_megabytes"
        ],
        "peak_rss_semantics": resource_report["peak_rss_semantics"],
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
    }


def benchmark_event_pipeline(
    *,
    iterations: int = 1000,
    frame_ms: int = 25,
) -> dict[str, object]:
    """Measure software event processing, excluding capture and XMOS DSP.

    The benchmark intentionally does not imply microphone-to-action latency or
    energy efficiency. It identifies the exact measured boundary in its report.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    geometry = ("left", "right")
    hierarchy = PredictiveHierarchy(
        frequency_coder=PredictiveCoder(300.0, learning_rate=0.2),
        focus_coder=FocusTrajectoryCoder(channels=geometry),
    )
    pipeline = CoraInfansPipeline(
        config=CoraInfansConfig(
            mic_channel_count=2,
            mic_channel_geometry=geometry,
            saliency_threshold=0.02,
            workspace_window_ms=250,
        ),
        predictive_hierarchy=hierarchy,
    )

    # Warm caches and imports outside the measured interval.
    pipeline.process_frame(AudioFrame(0, [0.03, 0.02], 300.0))
    frames = [
        AudioFrame(
            timestamp_ms=(index + 1) * frame_ms,
            channel_energies=[0.03 + (index % 3) * 0.001, 0.02],
            dominant_frequency_hz=300.0 + (index % 7) * 5.0,
        )
        for index in range(iterations)
    ]
    cora = _measure_processor(frames, pipeline.process_frame)
    conventional = _measure_processor(
        frames, ConventionalFrameBaseline().process_frame
    )
    represented_audio_seconds = iterations * frame_ms / 1000.0

    return {
        "mode": "event_pipeline_microbenchmark",
        "measurement_boundary": "AudioFrame_to_WorkspaceEvent_in_process",
        "excluded": [
            "microphone_capture",
            "xmos_dsp",
            "usb_transport",
            "wav_decoding",
            "speaker_playback",
            "energy_measurement",
        ],
        "iterations": iterations,
        "frame_ms": frame_ms,
        "latency_ms": cora["latency_ms"],
        "wall_seconds": cora["wall_seconds"],
        "cpu_seconds": cora["cpu_seconds"],
        "real_time_factor": float(cora["wall_seconds"]) / represented_audio_seconds,
        "average_process_cpu_percent": cora["average_process_cpu_percent"],
        "peak_resident_memory_bytes": cora["peak_resident_memory_bytes"],
        "peak_resident_memory_megabytes": cora["peak_resident_memory_megabytes"],
        "peak_rss_semantics": cora["peak_rss_semantics"],
        "tracemalloc_current_bytes": cora["tracemalloc_current_bytes"],
        "tracemalloc_peak_bytes": cora["tracemalloc_peak_bytes"],
        "resident_workspace_events": len(pipeline.workspace.snapshot()),
        "conventional_baseline": {
            "name": "fixed_threshold_nearest_frequency_bounded_deque",
            "measurement_boundary": "AudioFrame_to_fixed_DSP_event_in_process",
            **conventional,
            "real_time_factor": (
                float(conventional["wall_seconds"]) / represented_audio_seconds
            ),
        },
        "same_host_comparison": {
            "cora_to_baseline_p95_latency_ratio": (
                float(cora["latency_ms"]["p95"])
                / max(float(conventional["latency_ms"]["p95"]), 1e-12)
            ),
            "cora_to_baseline_real_time_factor_ratio": (
                float(cora["wall_seconds"])
                / max(float(conventional["wall_seconds"]), 1e-12)
            ),
            "input_frames_identical": True,
            "instrumentation_identical": True,
            "interpretation": (
                "The conventional baseline is intentionally simpler and is a "
                "resource floor, not a capability-equivalent competitor."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "pid": os.getpid(),
        },
    }
