#!/usr/bin/env python3
"""Run the frozen design-only Option E coefficient-family search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.factorized_tract_plant import evaluate_design_family


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = Path(args.protocol)
    report = evaluate_design_family()
    report["provenance"] = {
        "protocol": str(protocol),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "implementation_commit": args.implementation_commit,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
