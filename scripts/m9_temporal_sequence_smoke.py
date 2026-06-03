#!/usr/bin/env python3
"""M9 temporal-sequence smoke harness.

Runs the saved temporal controller/substrate pair on a short tone sequence and
prints a compact JSON report. This is hardware-free and suitable for regression
checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.temporal import run_temporal_smoke_report


def main() -> None:
    report = run_temporal_smoke_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
