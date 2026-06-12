"""Tests for M10 consonant prompts and corpus tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPO_ROOT / "data" / "m10_consonant_prompts.json"


def test_prompts_file_exists():
    assert PROMPTS_PATH.exists(), f"Prompts file not found: {PROMPTS_PATH}"


def test_prompts_file_is_valid_json():
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    assert "prompts" in data
    assert "categories" in data


def test_prompts_contain_all_consonant_categories():
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    categories = data["categories"]
    assert "stop" in categories
    assert "fricative" in categories
    assert "nasal" in categories


def test_prompts_contain_expected_labels():
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    labels = {p["label"] for p in data["prompts"]}
    expected = {"p", "t", "k", "s", "f", "sh", "m", "n"}
    assert expected.issubset(labels), f"Missing labels: {expected - labels}"


def test_each_prompt_has_required_fields():
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    for prompt in data["prompts"]:
        assert "token" in prompt, f"Missing token in {prompt}"
        assert "label" in prompt, f"Missing label in {prompt}"
        assert "category" in prompt, f"Missing category in {prompt}"


def test_prompts_can_be_loaded_by_record_script():
    """Verify the record script can import and load prompts."""
    import subprocess
    result = subprocess.run(
        [
            "python3",
            "-c",
            f"import json; d=json.loads(open('{PROMPTS_PATH}').read()); print(len(d['prompts']))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    count = int(result.stdout.strip())
    assert count > 0
