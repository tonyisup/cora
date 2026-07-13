#!/usr/bin/env python3
"""Development-only LPC-order sweep; never run on confirmation singers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from cora_infans.io.motor_plan import analyze_vowel
from cora_infans.io.wav_frames import read_pcm_wav


PATTERN = re.compile(r".*/(?P<singer>female\d+|male\d+)/.*/.*_(?P<vowel>[aeiou])\.wav$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--orders", default="8,10,12,14,16,18,20,24,28,32,36,40,48")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    observations = []
    for record in manifest["members"]:
        member = record["member"]
        match = PATTERN.fullmatch(member)
        if match is None:
            raise ValueError(member)
        path = Path(args.root).joinpath(*Path(member).parts)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"hash mismatch: {member}")
        samples, rate = read_pcm_wav(path)
        mono = np.mean(samples, axis=1)
        start = (len(mono) - rate) // 2
        observations.append((match.group("singer"), match.group("vowel"), mono[start : start + rate], rate))

    reports = {}
    for order in (int(value) for value in args.orders.split(",")):
        rows = []
        for singer, vowel, window, rate in observations:
            analysis = analyze_vowel(window, sample_rate=rate, order=order)
            rows.append((singer, vowel, analysis.nearest_vowel, analysis.formant_hz))
        reports[str(order)] = {
            "accuracy": sum(vowel == predicted for _, vowel, predicted, _ in rows) / len(rows),
            "rows": [
                {
                    "singer": singer,
                    "expected": vowel,
                    "predicted": predicted,
                    "formants_hz": list(formants),
                }
                for singer, vowel, predicted, formants in rows
            ],
        }
    selected = max(reports, key=lambda order: (reports[order]["accuracy"], -int(order)))
    print(json.dumps({"mode": "development_only_lpc_order_sweep", "reports": reports, "selected_order": int(selected), "selected_accuracy": reports[selected]["accuracy"], "confirmation_audio_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
