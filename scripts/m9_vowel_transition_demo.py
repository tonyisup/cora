#!/usr/bin/env python3
"""M9 vowel-transition listening demo.

Renders the new longer transition families to separate WAV files and a single
combined montage so you can listen through the progression in one pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))

from m9_sound.temporal import write_pcm16_wav
from m9_sound.vowel_transitions import VOWEL_FAMILIES, build_vowel_transition_report, write_vowel_transition_wav

DEFAULT_FAMILIES = ["long_arc", "formant_rich"]
DEFAULT_PAUSE_SECONDS = 0.35
SR = 16000


def _render_section(output_dir: Path, family: str, heldout_family: str) -> dict[str, object]:
    section_path = output_dir / f"m9-vowel-transition-{family}.wav"
    return write_vowel_transition_wav(
        section_path,
        family=family,
        heldout_family=heldout_family,
        noise_rms=0.0,
        noise_seed=7,
        device_profile="flat",
        device_gain=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 vowel-transition listening demo")
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
        choices=sorted(VOWEL_FAMILIES.keys()),
        help="Transition families to render into the listening montage",
    )
    parser.add_argument(
        "--heldout-family",
        type=str,
        default="diphthong",
        choices=sorted(VOWEL_FAMILIES.keys()),
        help="Held-out family used inside each report",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/cora-m9-vowel-transition-demo"),
        help="Directory for the per-family WAVs, montage WAV, and JSON manifest",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Silence inserted between families in the combined montage",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path; defaults to <output-dir>/m9-vowel-transition-demo.json",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (output_dir / "m9-vowel-transition-demo.json")

    sections: list[dict[str, object]] = []
    rendered_audio: list[np.ndarray] = []
    pause_samples = int(round(args.pause_seconds * SR))
    pause = np.zeros(max(0, pause_samples), dtype=np.float32)

    for idx, family in enumerate(args.families):
        report = _render_section(output_dir, family, args.heldout_family)
        sections.append(
            {
                "family": report["family"],
                "wav": report["output_wav"],
                "reward": report["reward"],
                "transition_segment_count": report["transition_segment_count"],
                "segment_frame_count": report["segment_frame_count"],
                "stimulus_frame_count": report["stimulus_frame_count"],
                "stimulus_samples": report["stimulus_samples"],
            }
        )
        rendered_audio.append(np.asarray(report["rendered_audio"], dtype=np.float32))
        if idx != len(args.families) - 1 and len(pause):
            rendered_audio.append(pause)

    montage = np.concatenate(rendered_audio) if rendered_audio else np.zeros(0, dtype=np.float32)
    montage_path = output_dir / "m9-vowel-transition-demo.wav"
    montage_meta = write_pcm16_wav(montage_path, montage, sr=SR)

    manifest: dict[str, object] = {
        "output_dir": str(output_dir),
        "families": list(args.families),
        "heldout_family": args.heldout_family,
        "pause_seconds": float(args.pause_seconds),
        "montage_wav": str(montage_meta["path"]),
        "montage_sample_count": int(montage_meta["sample_count"]),
        "sections": sections,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
