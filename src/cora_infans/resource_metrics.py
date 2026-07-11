"""Process-level CPU and resident-memory measurements with explicit boundaries."""

from __future__ import annotations

import platform
import resource
import time


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSDs conventionally report KiB.
    return value if platform.system() == "Darwin" else value * 1024


class ProcessResourceTracker:
    """Measure average process CPU and process-lifetime peak RSS around a block."""

    def __init__(self) -> None:
        self._wall_started = time.perf_counter()
        self._cpu_started = time.process_time()
        self._rss_started = _peak_rss_bytes()

    def summary(self) -> dict[str, object]:
        wall_seconds = time.perf_counter() - self._wall_started
        cpu_seconds = time.process_time() - self._cpu_started
        peak_rss_bytes = _peak_rss_bytes()
        return {
            "measurement_boundary": "current_process_over_measured_operation",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "average_process_cpu_percent": (
                cpu_seconds / wall_seconds * 100.0 if wall_seconds > 0.0 else 0.0
            ),
            "peak_resident_memory_bytes": peak_rss_bytes,
            "peak_resident_memory_megabytes": peak_rss_bytes / (1024.0 * 1024.0),
            "peak_rss_at_start_bytes": self._rss_started,
            "peak_rss_semantics": (
                "process_lifetime_high_water_mark_sampled_at_operation_end; "
                "not operation-attributable incremental memory"
            ),
            "cpu_scope": "current_process_all_threads_not_system_energy",
        }
