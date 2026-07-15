#!/usr/bin/env python3
"""Score a bounded vocal-imitation replay directory without live audio I/O."""

from __future__ import annotations

import argparse
import json

from cora_infans.product_session import EVIDENCE_CLASSES, score_product_session_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--evidence-class", required=True, choices=sorted(EVIDENCE_CLASSES))
    parser.add_argument("--association-frozen-before-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.association_frozen_before_evaluation:
        parser.error("--association-frozen-before-evaluation is required")
    report = score_product_session_directory(
        args.session_dir,
        evidence_class=args.evidence_class,
        association_frozen_before_evaluation=args.association_frozen_before_evaluation,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
