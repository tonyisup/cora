#!/usr/bin/env python3
"""Run the frozen H3 same-source forward-model development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cora_infans.same_source_forward_model import (
    evaluate_spent_same_source_forward_model,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-v2-result", required=True)
    parser.add_argument("--template-v2-artifacts", required=True)
    parser.add_argument("--spectral-v3-result", required=True)
    parser.add_argument("--spectral-v3-artifacts", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = evaluate_spent_same_source_forward_model(
        template_v2_result=args.template_v2_result,
        template_v2_artifacts=args.template_v2_artifacts,
        spectral_v3_result=args.spectral_v3_result,
        spectral_v3_artifacts=args.spectral_v3_artifacts,
    )
    preregistration = Path(args.preregistration)
    report["provenance"] = {
        "preregistration": str(preregistration),
        "preregistration_sha256": _sha256(preregistration),
        "preregistration_commit": "bc83cca",
        "implementation_commit": args.implementation_commit,
        "template_v2_result_sha256": _sha256(Path(args.template_v2_result)),
        "spectral_v3_result_sha256": _sha256(Path(args.spectral_v3_result)),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
