import time

from cora_infans.resource_metrics import ProcessResourceTracker


def test_process_resource_tracker_reports_explicit_scope() -> None:
    tracker = ProcessResourceTracker()
    time.sleep(0.001)
    report = tracker.summary()

    assert report["measurement_boundary"] == "current_process_over_measured_operation"
    assert report["wall_seconds"] > 0.0
    assert report["cpu_seconds"] >= 0.0
    assert report["average_process_cpu_percent"] >= 0.0
    assert report["peak_resident_memory_bytes"] > 0
    assert "process_lifetime_high_water_mark" in report["peak_rss_semantics"]
