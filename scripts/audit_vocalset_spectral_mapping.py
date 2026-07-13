#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from cora_infans.vocalset_spectral_mapping import audit_spectral_mapping_candidates


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--manifest", required=True)
args = parser.parse_args()
print(json.dumps(audit_spectral_mapping_candidates(corpus_root=args.root, manifest_path=args.manifest)))
