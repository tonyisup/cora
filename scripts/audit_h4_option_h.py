#!/usr/bin/env python3
"""Run the frozen H4 Option H smooth local-closure candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.smooth_local_closure import evaluate_option_h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    preregistration = Path(args.preregistration)
    report = evaluate_option_h()
    report["provenance"] = {
        "preregistration": str(preregistration),
        "preregistration_sha256": hashlib.sha256(
            preregistration.read_bytes()
        ).hexdigest(),
        "preregistration_commit": "22bc25f",
        "implementation_commit": args.implementation_commit,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
