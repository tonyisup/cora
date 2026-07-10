"""Reproducible microbenchmarks for the bounded event-processing core."""

from __future__ import annotations

import os
import platform
import statistics
import time
import tracemalloc

import numpy as np

from .config import CoraInfansConfig
from .pipeline import AudioFrame, CoraInfansPipeline
from .predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


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
    durations_ms: list[float] = []
    process_start = time.process_time()
    wall_start = time.perf_counter()
    tracemalloc.start()
    for index in range(iterations):
        frame = AudioFrame(
            timestamp_ms=(index + 1) * frame_ms,
            channel_energies=[0.03 + (index % 3) * 0.001, 0.02],
            dominant_frequency_hz=300.0 + (index % 7) * 5.0,
        )
        started = time.perf_counter_ns()
        pipeline.process_frame(frame)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - process_start
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
        "latency_ms": {
            "mean": statistics.fmean(durations_ms),
            "p50": _percentile(durations_ms, 50.0),
            "p95": _percentile(durations_ms, 95.0),
            "p99": _percentile(durations_ms, 99.0),
            "max": max(durations_ms),
        },
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "real_time_factor": wall_seconds / represented_audio_seconds,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "resident_workspace_events": len(pipeline.workspace.snapshot()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "pid": os.getpid(),
        },
    }
