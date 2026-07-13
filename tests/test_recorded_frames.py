from __future__ import annotations

import json

import numpy as np

from cora_infans.io.offline_eval import CorpusLoader, OfflineEvalHarness
from cora_infans.io.recorded_frames import (
    build_corollary_capture_manifest,
    build_recorded_npy_frames,
)


def _write_capture(path, *, seed: int, self_only: bool, world_only: bool) -> None:
    path.mkdir()
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    np.save(path / "observed.npy", 0.2 * np.sin(2 * np.pi * 220.0 * t))
    (path / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "self_only": self_only,
                "world_only": world_only,
            }
        )
    )


def test_prior_art_capture_manifest_and_provider_preserve_real_provenance(tmp_path) -> None:
    _write_capture(tmp_path / "self_seed0", seed=0, self_only=True, world_only=False)
    _write_capture(tmp_path / "mixed_seed2", seed=2, self_only=False, world_only=False)
    manifest = tmp_path / "labels.csv"

    summary = build_corollary_capture_manifest(tmp_path, manifest)
    clips = CorpusLoader(tmp_path).load_labels(manifest)
    report = OfflineEvalHarness(
        frame_ms=100,
        saliency_threshold=0.01,
    ).evaluate_corpus(clips, build_recorded_npy_frames)

    assert summary["clip_count"] == 2
    assert summary["dev_clip_count"] == 1
    assert summary["held_out_clip_count"] == 1
    assert {clip.sample_rate for clip in clips} == {16000}
    assert report["evidence_kind"] == "real_audio"
    assert {item["input_source"] for item in report["per_clip"]} == {"recorded_npy"}
    assert report["total_frames"] == 20
