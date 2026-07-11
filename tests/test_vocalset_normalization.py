import hashlib
import json
from pathlib import Path

import numpy as np

import cora_infans.vocalset_normalization as normalization
from cora_infans.io.motor_plan import VowelAnalysis, write_wav


def test_speaker_normalization_selects_transfer_candidate(monkeypatch, tmp_path) -> None:
    singers = ("female4", "female5", "male6", "male7", "male8")
    vowels = ("a", "e", "i", "o", "u")
    base = {
        "a": (800.0, 1200.0, 2500.0),
        "e": (600.0, 1900.0, 2500.0),
        "i": (300.0, 2400.0, 3000.0),
        "o": (650.0, 900.0, 2400.0),
        "u": (350.0, 950.0, 2200.0),
    }
    calls = []

    def analyze(_window, *, sample_rate):
        singer_index, vowel_index = divmod(len(calls), 5)
        vowel = vowels[vowel_index]
        scale = 0.75 + 0.15 * singer_index
        measured = tuple(scale * value + 40 * singer_index for value in base[vowel])
        calls.append((singer_index, vowel))
        return VowelAnalysis(measured, "e", 0.0)

    monkeypatch.setattr(normalization, "analyze_vowel", analyze)
    records = []
    for singer in singers:
        short = ("f" if singer.startswith("female") else "m") + "".join(
            character for character in singer if character.isdigit()
        )
        for vowel in vowels:
            member = f"FULL/{singer}/long_tones/straight/{short}_long_straight_{vowel}.wav"
            path = tmp_path / member
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(str(path), np.zeros(44100), 44100)
            records.append(
                {"member": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"members": records}))

    report = normalization.audit_normalization_candidates(
        corpus_root=tmp_path, manifest_path=manifest
    )

    assert report["reports"]["raw_fixed_table"]["accuracy"] == 0.2
    assert report["selected_report"]["accuracy"] >= 0.8
    assert report["selected_candidate"] != "raw_fixed_table"
    assert report["confirmation_audio_accessed"] is False
    assert report["promotion_allowed"] is False


def test_retained_development_ladder_preserves_stop_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence"
    audit = json.loads(
        (evidence / "2026-07-11-vocalset-normalization-development-audit.json").read_text()
    )
    normalization = json.loads(
        (evidence / "2026-07-11-vocalset-normalization-development-result.json").read_text()
    )
    lpc = json.loads(
        (evidence / "2026-07-11-vocalset-lpc-order-development-result.json").read_text()
    )
    spectral = json.loads(
        (evidence / "2026-07-11-vocalset-spectral-mapping-development-result.json").read_text()
    )

    assert audit["confirmation_audio_accessed"] is False
    assert audit["promotion_allowed"] is False
    assert normalization["selected_report"]["accuracy"] == 0.36
    assert lpc["selected_accuracy"] == 0.4
    assert spectral["selected_report"]["accuracy"] == 0.56
    assert spectral["selected_report"]["mapping"] == (
        "spectral_centroid_to_canonical_motor_target"
    )
    assert spectral["confirmation_audio_accessed"] is False
    assert spectral["promotion_allowed"] is False
