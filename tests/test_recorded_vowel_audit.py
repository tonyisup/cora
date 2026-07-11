import hashlib
import json

from cora_infans.io.motor_plan import synthesize_vowel, write_wav
from cora_infans.recorded_vowel_audit import (
    VOCALSET_CONFIRMATION_SINGERS,
    VOWELS,
    build_recorded_vowel_protocol,
    evaluate_recorded_vowels,
)


def _fixture(tmp_path):
    records = []
    for singer in VOCALSET_CONFIRMATION_SINGERS:
        short = ("f" if singer.startswith("female") else "m") + singer.lstrip(
            "female"
        ).lstrip("male")
        for index, vowel in enumerate(VOWELS):
            member = f"FULL/{singer}/long_tones/straight/{short}_long_straight_{vowel}.wav"
            path = tmp_path / member
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(
                str(path),
                synthesize_vowel(vowel, duration_ms=1200, sample_rate=44100, f0_hz=90 + 15 * index),
                44100,
            )
            records.append(
                {
                    "member": member,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": "VocalSet 1.0",
                "doi": "10.5281/zenodo.1193957",
                "license": "CC BY 4.0",
                "members": records,
            }
        )
    )
    return manifest


def test_recorded_vowel_protocol_freezes_held_out_scope() -> None:
    protocol = build_recorded_vowel_protocol()
    assert protocol["expected_clip_count"] == 25
    assert protocol["confirmation_singers"] == list(VOCALSET_CONFIRMATION_SINGERS)
    assert protocol["vowels"] == list(VOWELS)
    assert protocol["motor_learning_claim_allowed"] is False


def test_recorded_vowel_evaluator_reports_both_matrices_and_formant_errors(tmp_path) -> None:
    report = evaluate_recorded_vowels(
        corpus_root=tmp_path, manifest_path=_fixture(tmp_path)
    )

    assert report["verdict"] in {
        "recorded_vowel_gate_passed",
        "recorded_vowel_gate_failed",
    }
    assert report["promotion_allowed"] is all(report["criteria"].values())
    assert report["synthetic"]["trial_count"] == 40
    assert report["recorded"]["trial_count"] == 25
    assert set(report["recorded"]["formant_mae_hz"]) == {"f1", "f2", "f3"}
    assert all(report["criteria"].values())


def test_recorded_vowel_evaluator_rejects_hash_drift(tmp_path) -> None:
    manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["members"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))

    try:
        evaluate_recorded_vowels(corpus_root=tmp_path, manifest_path=manifest)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("corpus hash drift must fail")


def test_retained_vocalset_confirmation_preserves_falsification() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/evidence/2026-07-11-vocalset-recorded-vowel-result.json").read_text()
    )

    assert result["verdict"] == "recorded_vowel_gate_failed"
    assert result["promotion_allowed"] is False
    assert result["synthetic"]["accuracy"] == 1.0
    assert result["recorded"]["accuracy"] == 0.36
    assert result["recorded"]["vowel_recall"]["i"] == 0.0
    assert result["recorded"]["vowel_recall"]["u"] == 0.2
    assert result["criteria"]["all_member_hashes_verified"] is True
