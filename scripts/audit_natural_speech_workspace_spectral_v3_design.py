#!/usr/bin/env python3
"""Retain the frozen spectral-v3 design audit over spent template-v2 PCM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.natural_speech_workspace_spectral_v3 import audit_spent_v2_design


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit_spent_v2_design(
        artifacts=args.artifacts,
        result=args.result,
        corpus_root=args.corpus_root,
    )
    preregistration = Path(args.preregistration)
    report["preregistration"] = {
        "path": str(preregistration),
        "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "commit": args.preregistration_commit,
    }
    report["implementation_commit"] = args.implementation_commit
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
