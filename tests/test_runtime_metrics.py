from cora_infans.runtime_metrics import RuntimeLatencyTracker


def test_runtime_latency_tracker_reports_deadlines_and_boundary() -> None:
    tracker = RuntimeLatencyTracker(deadline_ms=25.0)
    tracker.record(capture_ms=20.0, pipeline_ms=1.0, total_ms=21.0)
    tracker.record(capture_ms=30.0, pipeline_ms=1.0, total_ms=31.0)

    report = tracker.summary(
        capture_backend="FakeArray",
        real_capture=False,
        backend_reported_input_latency_ms=10.0,
        input_overflow_count=1,
    )

    assert report["measurement_boundary"] == "capture_frame_call_start_to_workspace_decision"
    assert report["frame_count"] == 2
    assert report["deadline_miss_count"] == 1
    assert report["deadline_miss_rate"] == 0.5
    assert report["strict_deadline_miss_count"] == 1
    assert report["deadline_tolerance_ms"] == 5.0
    assert report["requested_frame_rate_hz"] == 40.0
    assert report["measured_path_real_time_factor"] == 52.0 / 50.0
    assert report["cold_start_capture_to_workspace_ms"] == 21.0
    assert report["steady_state_real_time_factor"] == 31.0 / 25.0
    assert report["capture_to_workspace_latency_ms"]["p95"] >= 21.0
    assert report["real_capture"] is False
    assert report["backend_reported_input_latency_ms"] == 10.0
    assert report["estimated_adc_to_workspace_latency_ms"]["p50"] == 36.0
    assert report["input_overflow_count"] == 1
    assert report["input_overflow_rate"] == 0.5


def test_runtime_tracker_reports_device_timestamped_frame_latency() -> None:
    tracker = RuntimeLatencyTracker(deadline_ms=25.0)
    tracker.record(
        capture_ms=25.0,
        pipeline_ms=1.0,
        total_ms=26.0,
        adc_onset_to_decision_ms=51.0,
        adc_end_to_decision_ms=26.0,
        workload_label="salient",
    )

    report = tracker.summary(
        capture_backend="sounddevice_portaudio_coreaudio_callback",
        real_capture=True,
    )

    device = report["device_timestamped_frame_latency_ms"]
    assert report["timestamp_quality"] == "portaudio_inputBufferAdcTime_to_stream_time"
    assert device["adc_frame_onset_to_workspace_decision"]["p95"] == 51.0
    assert device["adc_frame_end_to_workspace_decision"]["p95"] == 26.0
    assert report["device_timestamped_strict_deadline_miss_count"] == 1
    assert report["device_timestamped_deadline_miss_count"] == 0
    assert "hardware_adc_sample_onset" not in report["excluded"]
    assert "raw_coreaudio_AudioTimeStamp" in report["excluded"]
    assert report["workload_latency_ms"]["salient"]["frame_count"] == 1
    assert report["workload_latency_ms"]["salient"][
        "adc_frame_end_to_workspace_decision"
    ]["p95"] == 26.0
