#!/usr/bin/env python3
"""Retain the preregistered spent template-v2 workspace design result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.natural_speech_workspace_template_v2 import (
    run_spent_template_design_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    result = run_spent_template_design_gate(args.root)
    preregistration = Path(args.preregistration)
    result["preregistration"] = {
        "path": str(preregistration),
        "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "commit": args.preregistration_commit,
    }
    result["implementation_commit"] = args.implementation_commit
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
