from __future__ import annotations

import json
from pathlib import Path


def test_data_dictionary_covers_common_acronyms_and_official_fieldsite_names() -> None:
    root = Path(__file__).resolve().parents[1]
    dictionary = (root / "docs/DATA_DICTIONARY.md").read_text()

    for term in (
        "ADS",
        "IDS",
        "F1 score",
        "F1/F2/F3",
        "FPR",
        "LOFO",
        "RMS",
        "ADC",
        "CORAL",
        "RBF",
        "RTF",
        "RSS",
        "XVF3800",
        "Product-session cohort",
        "blocked_pre_io",
        "technical_invalid",
        "behavioral_failed",
        "Behavioral repeatability",
        "Operational reliability",
        "owner_approved_not_executable",
    ):
        assert term in dictionary

    assert "`MEN` | Mentawai Islanders" in dictionary
    assert "`MES` | Colombian Mestizos" in dictionary
    assert "`A` = infant-directed song" in dictionary
    assert "`D` = ADS" in dictionary


def test_fieldsite_name_correction_preserves_data_and_metric_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    correction = json.loads(
        (
            root / "docs/evidence/2026-07-11-fieldsite-name-correction.json"
        ).read_text()
    )
    mapping = {
        item["fieldsite_code"]: item["official_name"]
        for item in correction["corrections"]
    }

    assert mapping == {
        "MEN": "Mentawai Islanders",
        "MES": "Colombian Mestizos",
    }
    assert correction["metric_impact"] == "none"
    assert correction["data_split_impact"] == "none"
    assert correction["protected_data_impact"] == "none"
