"""Sustained equivalent-input workloads for external power measurement."""

from __future__ import annotations

import platform
import time

from .benchmark import ConventionalFrameBaseline
from .config import CoraInfansConfig
from .pipeline import AudioFrame, CoraInfansPipeline
from .predictive import FocusTrajectoryCoder, PredictiveCoder, PredictiveHierarchy
from .resource_metrics import ProcessResourceTracker


def _cora_pipeline() -> CoraInfansPipeline:
    geometry = ("left", "right")
    hierarchy = PredictiveHierarchy(
        frequency_coder=PredictiveCoder(300.0, learning_rate=0.2),
        focus_coder=FocusTrajectoryCoder(channels=geometry),
    )
    return CoraInfansPipeline(
        config=CoraInfansConfig(
            mic_channel_count=2,
            mic_channel_geometry=geometry,
            saliency_threshold=0.02,
            workspace_window_ms=250,
        ),
        predictive_hierarchy=hierarchy,
    )


def _frames() -> list[AudioFrame]:
    return [
        AudioFrame(
            timestamp_ms=index * 25,
            channel_energies=[0.07 + (index % 3) * 0.001, 0.01],
            dominant_frequency_hz=300.0 + (index % 7) * 100.0,
        )
        for index in range(256)
    ]


def run_sustained_energy_workload(
    *, mode: str, duration_seconds: float
) -> dict[str, object]:
    """Run a CPU-bound workload for an external same-host power sampler."""

    if mode not in {"cora", "conventional"}:
        raise ValueError("mode must be cora or conventional")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    cora = _cora_pipeline() if mode == "cora" else None
    conventional = ConventionalFrameBaseline() if mode == "conventional" else None
    processor = (
        cora.process_frame if cora is not None else conventional.process_frame  # type: ignore[union-attr]
    )
    frames = _frames()
    for frame in frames:
        processor(frame)

    tracker = ProcessResourceTracker()
    deadline = time.perf_counter() + duration_seconds
    processed = 0
    emitted = 0
    checksum = 0
    while time.perf_counter() < deadline:
        for template in frames:
            frame = AudioFrame(
                timestamp_ms=(processed + len(frames)) * 25,
                channel_energies=template.channel_energies,
                dominant_frequency_hz=template.dominant_frequency_hz,
            )
            output = processor(frame)
            processed += 1
            if output is not None:
                emitted += 1
                checksum = (checksum + processed) % 2_147_483_647
            if time.perf_counter() >= deadline:
                break
    resources = tracker.summary()
    wall_seconds = float(resources["wall_seconds"])
    return {
        "mode": f"{mode}_sustained_energy_workload",
        "processor": mode,
        "measurement_boundary": (
            "monotonic_AudioFrame_construction_to_processor_output"
        ),
        "duration_requested_seconds": duration_seconds,
        "wall_seconds": wall_seconds,
        "frames_processed": processed,
        "events_emitted": emitted,
        "frames_per_second": processed / wall_seconds,
        "output_checksum": checksum,
        "resident_event_count": (
            len(cora.workspace.snapshot())
            if cora is not None
            else len(conventional.events)  # type: ignore[union-attr]
        ),
        "average_process_cpu_percent": resources["average_process_cpu_percent"],
        "peak_resident_memory_megabytes": resources[
            "peak_resident_memory_megabytes"
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "external_measurement_required": {
            "tool": "macOS powermetrics",
            "metric": "same-host estimated CPU/SoC power",
            "not_measured_by_this_report": True,
        },
        "excluded": [
            "microphone_capture",
            "speaker_playback",
            "xmos_dsp",
            "cross_device_power_comparison",
        ],
    }
