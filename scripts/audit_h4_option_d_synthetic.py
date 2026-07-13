#!/usr/bin/env python3
"""Run the frozen H4 Option D synthetic-only evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.geometric_articulatory_synthesis import evaluate_option_d_synthetic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preregistration = Path(args.preregistration)
    report = evaluate_option_d_synthetic()
    report["provenance"] = {
        "preregistration": str(preregistration),
        "preregistration_sha256": _sha256(preregistration),
        "preregistration_commit": "0c08f9a",
        "implementation_commit": args.implementation_commit,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
