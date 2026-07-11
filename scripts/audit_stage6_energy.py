#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from cora_infans.energy_audit import audit_energy_trials


parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()
print(json.dumps(audit_energy_trials(args.root), indent=2))
