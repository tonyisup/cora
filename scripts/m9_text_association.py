#!/usr/bin/env python3
"""M9d — Text association (word-sound pairing).

This is a synthetic text-to-sound wiring check. It creates word nodes and
acoustic prototype nodes, connects them with bidirectional SAG edges, and
verifies that both directions can be retrieved reliably.

Live audio end-to-end remains a later promotion step; this script is the
current M9d synthetic gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m9_sound import TextAssociationStore


DEFAULT_PAIRS: list[tuple[str, float, str]] = [
    ("alpha", 220.0, "sine"),
    ("beta", 247.0, "triangle"),
    ("gamma", 262.0, "sawtooth"),
    ("delta", 294.0, "square"),
    ("epsilon", 330.0, "sine"),
    ("zeta", 349.0, "triangle"),
    ("eta", 392.0, "sawtooth"),
    ("theta", 440.0, "square"),
    ("iota", 494.0, "sine"),
    ("kappa", 523.0, "triangle"),
]


def run_association(pairs: list[tuple[str, float, str]]) -> dict[str, Any]:
    store = TextAssociationStore()
    created = store.add_pairs(pairs)
    stats = store.evaluate()

    rows = []
    for pair in created:
        sound = store.get_sound_for_word(pair.word)
        word = store.get_word_for_sound(pair.sound_id)
        rows.append(
            {
                "word": pair.word,
                "word_id": pair.word_id,
                "sound_id": pair.sound_id,
                "freq_hz": pair.freq_hz,
                "waveform": pair.waveform,
                "word_to_sound": sound.id if sound is not None else None,
                "sound_to_word": word.id if word is not None else None,
                "word_to_sound_ok": sound is not None and sound.id == pair.sound_id,
                "sound_to_word_ok": word is not None and word.id == pair.word_id,
            }
        )

    return {
        "pairs": rows,
        "summary": {
            "total_pairs": stats.total_pairs,
            "word_to_sound_hits": stats.word_to_sound_hits,
            "sound_to_word_hits": stats.sound_to_word_hits,
            "word_to_sound_accuracy": stats.word_to_sound_accuracy,
            "sound_to_word_accuracy": stats.sound_to_word_accuracy,
            "all_passed": stats.all_passed,
            "graph_nodes": len(store.graph.nodes()),
            "graph_edges": len(store.graph.edges()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9d text association gate")
    parser.add_argument("--output", type=str, default="/tmp/m9_text_association.json")
    parser.add_argument("--pairs", type=int, default=len(DEFAULT_PAIRS))
    args = parser.parse_args()

    pairs = DEFAULT_PAIRS[: max(1, min(args.pairs, len(DEFAULT_PAIRS)))]

    print("=" * 72)
    print("M9d — Text Association")
    print("=" * 72)
    print(f"  Pairs: {len(pairs)}")
    print("  Live audio end-to-end: deferred to a later promotion gate")
    print()

    result = run_association(pairs)
    summary = result["summary"]

    for row in result["pairs"]:
        print(
            f"  {row['word']:<8} -> {row['sound_to_word']:<16} "
            f"word_to_sound={row['word_to_sound_ok']} sound_to_word={row['sound_to_word_ok']}"
        )

    print()
    print("Summary")
    print("-" * 72)
    print(f"  word->sound accuracy: {summary['word_to_sound_accuracy']:.2f}")
    print(f"  sound->word accuracy: {summary['sound_to_word_accuracy']:.2f}")
    print(f"  graph nodes:          {summary['graph_nodes']}")
    print(f"  graph edges:          {summary['graph_edges']}")
    print(f"  Verdict:              {'PASS' if summary['all_passed'] and len(pairs) >= 10 else 'FAIL'}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"config": {"pairs": len(pairs)}, **result}, indent=2))
    print(f"  Saved to:             {args.output}")

    if not (summary["all_passed"] and len(pairs) >= 10):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
