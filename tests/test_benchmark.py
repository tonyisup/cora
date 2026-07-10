from cora_infans.benchmark import benchmark_event_pipeline


def test_event_pipeline_benchmark_reports_explicit_boundary() -> None:
    report = benchmark_event_pipeline(iterations=20, frame_ms=25)

    assert report["measurement_boundary"] == "AudioFrame_to_WorkspaceEvent_in_process"
    assert report["iterations"] == 20
    assert report["latency_ms"]["p95"] >= 0.0
    assert report["real_time_factor"] >= 0.0
    assert report["tracemalloc_peak_bytes"] >= report["tracemalloc_current_bytes"]
    assert "microphone_capture" in report["excluded"]
    assert "energy_measurement" in report["excluded"]
    assert report["resident_workspace_events"] <= 11
