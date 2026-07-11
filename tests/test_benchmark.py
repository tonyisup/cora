from cora_infans.benchmark import benchmark_event_pipeline
from cora_infans.benchmark import ConventionalFrameBaseline
from cora_infans.pipeline import AudioFrame


def test_conventional_baseline_uses_same_saliency_gate() -> None:
    baseline = ConventionalFrameBaseline()

    assert baseline.process_frame(AudioFrame(0, [0.03, 0.02], 300.0)) is None
    assert baseline.process_frame(AudioFrame(25, [0.07, 0.01], 500.0)) == (0, 1)


def test_event_pipeline_benchmark_reports_explicit_boundary() -> None:
    report = benchmark_event_pipeline(iterations=20, frame_ms=25)

    assert report["measurement_boundary"] == "AudioFrame_to_WorkspaceEvent_in_process"
    assert report["iterations"] == 20
    assert report["latency_ms"]["p95"] >= 0.0
    assert report["real_time_factor"] >= 0.0
    assert report["tracemalloc_peak_bytes"] >= report["tracemalloc_current_bytes"]
    assert report["peak_resident_memory_bytes"] > 0
    assert report["average_process_cpu_percent"] >= 0.0
    assert report["conventional_baseline"]["real_time_factor"] >= 0.0
    assert report["same_host_comparison"]["input_frames_identical"] is True
    assert report["same_host_comparison"]["instrumentation_identical"] is True
    assert "microphone_capture" in report["excluded"]
    assert "energy_measurement" in report["excluded"]
    assert report["resident_workspace_events"] <= 11
