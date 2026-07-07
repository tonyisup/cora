# Project Cora-Infans Kickoff Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: turn the current vision-only repository into a runnable, test-backed research scaffold for audio-centric infant-inspired learning.

Architecture: start with a smallest useful vertical slice rather than pretending the whole cognitive stack exists. Build a deterministic audio-event pipeline first: input frame -> acoustic saliency gate -> predictive-state update -> event workspace record. Keep hardware, learning, and multi-mic fusion behind interfaces so the first slice is runnable on a laptop without the ReSpeaker board.

Tech Stack: Python 3.13, pytest, dataclasses, stdlib JSON/typing/pathlib.

---

## Current baseline

Present files:
- `VISION.md` — project thesis and architecture intent

Missing today:
- package scaffold
- dependency metadata
- tests
- executable entrypoint
- implementation docs beyond the vision statement

## Phase 0 — foundation and repo hygiene

### Task 1: add front-door docs
Objective: give the repo a truthful README that explains current status and first vertical slice.
Files:
- Create: `README.md`
- Reference: `VISION.md`
Verification:
- `read_file README.md`
- README states what exists now vs what is planned

### Task 2: add Python package metadata
Objective: make the repo installable and testable.
Files:
- Create: `pyproject.toml`
Verification:
- `python3 -m pytest --version`
- `python3 -m pytest tests -q` runs once tests exist

## Phase 1 — smallest useful vertical slice

### Task 3: define a red test for saliency gating
Objective: encode the core behavior that an input frame only enters the workspace when acoustic saliency crosses threshold.
Files:
- Create: `tests/test_pipeline.py`
Verification:
- `python3 -m pytest tests/test_pipeline.py::test_pipeline_emits_workspace_event_when_salient -v`
- Expected first run: FAIL because module does not exist

### Task 4: implement minimal pipeline objects
Objective: add the first runnable model of the architecture.
Files:
- Create: `src/cora_infans/__init__.py`
- Create: `src/cora_infans/pipeline.py`
Verification:
- `python3 -m pytest tests/test_pipeline.py::test_pipeline_emits_workspace_event_when_salient -v`
- `python3 -m pytest tests/test_pipeline.py -q`

### Task 5: add a red test for a user-facing architecture summary
Objective: make the repo self-describing from the CLI.
Files:
- Create: `tests/test_cli.py`
Verification:
- `python3 -m pytest tests/test_cli.py::test_describe_command_prints_json_summary -v`
- Expected first run: FAIL before CLI implementation

### Task 6: implement the CLI
Objective: expose the first vertical slice through a small executable surface.
Files:
- Create: `src/cora_infans/cli.py`
- Modify: `pyproject.toml`
Verification:
- `python3 -m pytest tests/test_cli.py -q`
- `python3 -m cora_infans --describe`

## Phase 2 — configuration and experiment boundaries

### Task 7: add explicit configuration object
Objective: keep thresholds and timing assumptions inspectable instead of hardcoded magic.
Files:
- Create: `src/cora_infans/config.py`
- Create: `tests/test_config.py`
Verification:
- red: focused config test fails first
- green: focused config test passes

### Task 8: document module responsibilities
Objective: make the next implementation phase obvious.
Files:
- Create: `docs/architecture/phase-1-foundation.md`
Verification:
- doc names each planned subsystem and its owning module path

## Phase 3 — next slice after kickoff

### Task 9: add synthetic multi-mic input adapter
Objective: represent the 4-mic array contract without depending on live hardware yet.
Files:
- Create: `src/cora_infans/io/mic_array.py`
- Create: `tests/test_mic_array.py`
Verification:
- deterministic fixture-based pytest slice passes

### Task 10: add predictive-state residual update
Objective: move beyond a pure threshold gate into a minimal predictive-coding loop.
Files:
- Create: `src/cora_infans/predictive.py`
- Create: `tests/test_predictive.py`
Verification:
- residual update tests pass
- pipeline integration test still passes

## Implementation order

1. README
2. pyproject
3. red pipeline test
4. minimal pipeline implementation
5. red CLI test
6. CLI implementation
7. config object
8. architecture doc
9. synthetic multi-mic adapter
10. predictive residual module

## Non-negotiable constraints

- No fake milestone claims: README must distinguish vision from implemented code.
- Keep the first slice hardware-agnostic; ReSpeaker integration comes after deterministic local tests exist.
- Tests first for code changes.
- Thresholds and timing live in config, not magic numbers scattered across modules.
- Each public module must have a narrow responsibility and a deterministic smoke path.

## First-slice recommendation

Implement only the deterministic saliency-to-workspace loop in this kickoff. That gives us a real package, a real CLI, and real tests without overcommitting to the predictive-coding or motor-synthesis design before the boundaries are stable.
