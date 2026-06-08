#!/usr/bin/env python3
"""Migrate the legacy M9 temporal controller checkpoint to a producer checkpoint.

Reads:
  data/m9_controller_temporal_v3.ckpt.json
  data/m9_substrate_live.ckpt.json

Writes:
  data/m9_controller_temporal_v3_migrated.ckpt.json
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_SRC = PROJECT_ROOT / "data" / "m9_controller_temporal_v3.ckpt.json"
SUBSTRATE_SRC = PROJECT_ROOT / "data" / "m9_substrate_live.ckpt.json"
DEST = PROJECT_ROOT / "data" / "m9_controller_temporal_v3_migrated.ckpt.json"


def main() -> None:
    if not CONTROLLER_SRC.exists():
        raise FileNotFoundError(f"Missing controller source: {CONTROLLER_SRC}")
    if not SUBSTRATE_SRC.exists():
        raise FileNotFoundError(f"Missing substrate source: {SUBSTRATE_SRC}")

    controller_data = json.loads(CONTROLLER_SRC.read_text(encoding="utf-8"))
    substrate_data = json.loads(SUBSTRATE_SRC.read_text(encoding="utf-8"))

    dest = {
        "controller": controller_data,
        "substrate": substrate_data,
        "config": {
            "sr": 16000,
            "hidden_dim": 32,
            "frame_size": 480,
        },
    }

    DEST.write_text(json.dumps(dest, indent=2), encoding="utf-8")
    print(f"Wrote migrated producer checkpoint: {DEST}")


if __name__ == "__main__":
    main()
