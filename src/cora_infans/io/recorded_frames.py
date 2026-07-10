"""Adapters for real captured arrays retained by prior-art experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .offline_eval import CorpusClip, FrameObservation
from .wav_frames import build_pcm_observations


def build_recorded_npy_frames(
    clip: CorpusClip,
    frame_ms: int,
) -> list[FrameObservation]:
    if clip.sample_rate is None:
        raise ValueError(f"recorded NPY clip {clip.clip_id} requires sample_rate")
    samples = np.load(clip.wav_path, allow_pickle=False)
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError(f"recorded NPY clip {clip.clip_id} is not numeric PCM")
    return build_pcm_observations(
        np.asarray(samples, dtype=np.float32),
        sample_rate=clip.sample_rate,
        frame_ms=frame_ms,
        source="recorded_npy",
    )


def build_corollary_capture_manifest(
    root: str | Path,
    output_csv: str | Path,
    *,
    sample_rate: int = 16000,
    held_out_seeds: tuple[int, ...] = (2,),
) -> dict[str, object]:
    """Index archived Corollary observed.npy captures without copying audio."""

    import csv

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"capture root is not a directory: {root_path}")
    rows: list[dict[str, str]] = []
    for observed_path in sorted(root_path.rglob("observed.npy")):
        summary_path = observed_path.with_name("summary.json")
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        seed = int(summary.get("seed", -1))
        if summary.get("self_only") is True:
            label = "self_voice"
        elif summary.get("world_only") is True:
            label = "world_voice"
        else:
            label = "mixed_voice"
        samples = np.load(observed_path, mmap_mode="r", allow_pickle=False)
        duration_s = len(samples) / float(sample_rate)
        relative = observed_path.relative_to(root_path)
        clip_id = "__".join(relative.with_suffix("").parts)
        rows.append(
            {
                "clip_id": clip_id,
                "start_s": "0.0",
                "end_s": f"{duration_s:.6f}",
                "label": label,
                "split": "held_out" if seed in held_out_seeds else "dev",
                "source_path": str(relative),
                "sample_rate": str(sample_rate),
                "seed": str(seed),
            }
        )
    if not rows:
        raise ValueError(f"no observed.npy + summary.json pairs found under {root_path}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "start_s",
        "end_s",
        "label",
        "split",
        "source_path",
        "sample_rate",
        "seed",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "mode": "corollary_capture_manifest_builder",
        "root": str(root_path),
        "output_csv": str(output_path.resolve()),
        "clip_count": len(rows),
        "dev_clip_count": sum(row["split"] == "dev" for row in rows),
        "held_out_clip_count": sum(row["split"] == "held_out" for row in rows),
        "sample_rate": sample_rate,
        "held_out_seeds": list(held_out_seeds),
        "labels": sorted({row["label"] for row in rows}),
        "evidence_scope": "prior_art_real_captured_audio",
    }
