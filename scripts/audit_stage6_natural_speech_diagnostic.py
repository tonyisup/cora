#!/usr/bin/env python3
"""Audit three per-frame-instrumented natural-speech live reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cora_infans.natural_speech_timing_audit import (
    audit_natural_speech_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit_natural_speech_diagnostic(args.reports)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
