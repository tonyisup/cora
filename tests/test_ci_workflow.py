"""Governance checks for the hardware-safe CI workflow."""

from pathlib import Path

import yaml


def test_ci_runs_readiness_before_tests_with_read_only_permissions() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/ci.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}

    verify = workflow["jobs"]["verify"]
    assert verify["timeout-minutes"] == "15"
    actions = [step["uses"] for step in verify["steps"] if "uses" in step]
    assert actions == [
        "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    ]
    setup = next(
        step
        for step in verify["steps"]
        if "setup-python" in step.get("uses", "")
    )
    assert setup["with"]["python-version"] == "3.13.14"
    commands = [step["run"] for step in verify["steps"] if "run" in step]
    assert "--constraint requirements-ci.txt" in commands[0]
    assert commands.index(
        "python -m cora_infans --operational-readiness"
    ) < commands.index("python -m pytest tests -q")
    assert not any("--capture-live" in command for command in commands)
    assert not any("--real-io" in command for command in commands)
