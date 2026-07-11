#!/usr/bin/env python3
"""Run the frozen H4 continuous inverse-synthesis development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.continuous_inverse_synthesis import (
    evaluate_continuous_inverse_synthesis,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocalset-root", required=True)
    parser.add_argument("--vocalset-manifest", required=True)
    parser.add_argument("--cmu-root", required=True)
    parser.add_argument("--cmu-manifest", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preregistration = Path(
        "docs/evidence/"
        "2026-07-11-h4-continuous-inverse-synthesis-development-"
        "preregistration.json"
    )
    clarification = Path(
        "docs/evidence/"
        "2026-07-11-h4-continuous-inverse-synthesis-preregistration-"
        "clarification.json"
    )
    report = evaluate_continuous_inverse_synthesis(
        vocalset_root=args.vocalset_root,
        vocalset_manifest=args.vocalset_manifest,
        cmu_root=args.cmu_root,
        cmu_manifest=args.cmu_manifest,
    )
    report["provenance"] = {
        "implementation_commit": args.implementation_commit,
        "preregistration": str(preregistration),
        "preregistration_sha256": _sha256(preregistration),
        "clarification": str(clarification),
        "clarification_sha256": _sha256(clarification),
        "vocalset_manifest_sha256": _sha256(Path(args.vocalset_manifest)),
        "cmu_manifest_sha256": _sha256(Path(args.cmu_manifest)),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
