import hashlib
import json

import numpy as np

import cora_infans.cmu_arctic_audit as audit_module
from cora_infans.io.motor_plan import VowelAnalysis, write_wav


def test_cmu_arctic_audit_selects_label_blind_candidate(monkeypatch, tmp_path) -> None:
    speakers = ("bdl", "slt", "clb", "jmk", "ksp")
    vowels = ("a", "e", "i", "o", "u")
    calls = []

    def analyze(_window, *, sample_rate, order):
        vowel = vowels[len(calls) % 5]
        calls.append((sample_rate, order))
        return VowelAnalysis((700.0, 1200.0, 2500.0), vowel, 0.0)

    monkeypatch.setattr(audit_module, "analyze_vowel", analyze)
    tokens = []
    wav_files = []
    for speaker in speakers:
        for token_index in range(100):
            vowel = vowels[token_index % 5]
            name = f"arctic_a{token_index:04d}.wav"
            path = tmp_path / speaker / "wav" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(str(path), np.zeros(3200), 16000)
            wav_files.append(
                {
                    "speaker": speaker,
                    "name": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            tokens.append(
                {
                    "speaker": speaker,
                    "wav_name": name,
                    "segment_index": 0,
                    "vowel": vowel,
                    "start_seconds": 0.02,
                    "end_seconds": 0.18,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "confirmation_audio_or_labels_accessed": False,
                "tokens": tokens,
                "wav_files": wav_files,
            }
        )
    )

    report = audit_module.audit_cmu_arctic_development(
        corpus_root=tmp_path, manifest_path=manifest
    )

    assert report["token_count"] == 500
    assert report["selected_report"]["accuracy"] == 1.0
    assert report["confirmation_audio_or_labels_accessed"] is False
    assert report["promotion_allowed"] is False


def test_retained_cmu_arctic_development_result_preserves_stop_decision() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-cmu-arctic-development-result.json"
        ).read_text()
    )

    assert result["status"] == "development_candidate_failed_confirmation_readiness"
    assert result["token_count"] == 500
    assert result["selected_candidate"] == "window_120ms_order_20"
    assert result["selected_report"]["accuracy"] == 0.572
    assert result["selected_report"]["vowel_recall"]["o"] == 0.13
    assert result["selected_report"]["vowel_recall"]["u"] == 0.3
    assert result["confirmation_audio_or_labels_accessed"] is False
    assert result["promotion_allowed"] is False
