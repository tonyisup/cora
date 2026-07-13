#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from cora_infans.cmu_arctic_audit import audit_cmu_arctic_development


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--manifest", required=True)
args = parser.parse_args()
print(
    json.dumps(
        audit_cmu_arctic_development(
            corpus_root=args.root, manifest_path=args.manifest
        )
    )
)
