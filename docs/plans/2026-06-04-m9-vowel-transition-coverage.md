# M9 Vowel-Transition Coverage Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Extend M9 from pure-tone temporal coverage to deterministic vowel-transition family coverage over the spectral substrate path, with a smoke harness, held-out generalization checks, and listenable artifacts.

**Architecture:**
Build a small reusable package module for vowel-transition sequence generation and scoring, then layer a thin CLI smoke harness on top of it. Keep the runtime path in-memory and deterministic: generate a transition family, run it through the spectral substrate/controller path, compute continuity + transition metrics, and expose the result as JSON plus an optional WAV artifact. Add tests first for the reusable helper, then for the CLI wiring and artifact shape.

**Tech Stack:** Python, NumPy, PyTest, stdlib `wave`, existing `m9_sound` package, existing spectral checkpoints (`data/m9_substrate_spectral.ckpt.json`, `data/m9_formant_lookup_v5.json`).

---

### Task 1: Add deterministic vowel-transition family helpers

**Objective:** Create a reusable module that can generate and score small vowel-transition families deterministically.

**Files:**
- Create: `packages/m9_sound/vowel_transitions.py`
- Modify: `packages/m9_sound/__init__.py`
- Test: `tests/unit/test_m9_vowel_transitions.py`

**Step 1: Write failing tests**

Add tests that assert:
- a default vowel-transition family is deterministic and returns the same token sequence every run
- a longer family has the expected frame count and transition count
- a held-out family produces a different reward surface from the primary family
- the helper exposes a JSONable report with fields like `family`, `input_transition_count`, `decoded_transition_count`, `mean_boundary_delta`, and `heldout_family`

Example test shape:
```python
def test_default_family_is_deterministic():
    report = build_vowel_transition_report(family="default", noise_rms=0.0, noise_seed=7)
    assert report["family"] == "default"
    assert report["input_transition_count"] == 2
    assert report["decoded_transition_count"] == 2
```

**Step 2: Run the tests and verify failure**

Run:
```bash
.venv/bin/python -m pytest tests/unit/test_m9_vowel_transitions.py -q
```
Expected: fail because the module/helper does not exist yet.

**Step 3: Implement the minimal helper**

Add a small family table and helper functions such as:
```python
def build_vowel_transition_report(...):
    ...
```
The helper should:
- generate deterministic transition sequences
- use the existing spectral substrate path
- compute decode/continuity metrics
- return a JSONable dict

**Step 4: Run the tests to verify pass**

Run:
```bash
.venv/bin/python -m pytest tests/unit/test_m9_vowel_transitions.py -q
```
Expected: pass.

**Step 5: Commit**

```bash
git add packages/m9_sound/vowel_transitions.py packages/m9_sound/__init__.py tests/unit/test_m9_vowel_transitions.py
git commit -m "feat(m9): add vowel transition family helpers"
```

---

### Task 2: Add a thin vowel-transition smoke CLI

**Objective:** Expose the reusable helper as a CLI that prints structured JSON and can optionally emit a listenable WAV.

**Files:**
- Create: `scripts/m9_vowel_transition_smoke.py`
- Modify: `packages/m9_sound/vowel_transitions.py` (if needed for WAV export plumbing)
- Test: `tests/unit/test_m9_vowel_transitions.py`

**Step 1: Write failing tests**

Add tests that assert:
- the CLI-compatible helper writes a WAV file when requested
- the WAV file is mono 16-bit PCM at 16 kHz
- the report includes the exact family/noise/device settings used
- the longer family variant still preserves transition count under mild noise

Example command expectation:
```bash
.venv/bin/python scripts/m9_vowel_transition_smoke.py --family default --output-wav /tmp/m9-vowel-transition.wav
```
Expected: JSON output plus a valid WAV file path.

**Step 2: Run the tests and verify failure**

Run the focused pytest and the CLI once the script exists:
```bash
.venv/bin/python -m pytest tests/unit/test_m9_vowel_transitions.py -q
```
Expected: fail until the CLI wiring exists.

**Step 3: Implement the minimal CLI**

Keep the script thin:
- parse flags
- call the reusable helper
- print JSON
- optionally write WAV

Suggested flags:
- `--family`
- `--noise-rms`
- `--noise-seed`
- `--device-profile`
- `--device-gain`
- `--output-wav`
- `--output`

**Step 4: Run the CLI smoke**

Run:
```bash
.venv/bin/python scripts/m9_vowel_transition_smoke.py --family default --output /tmp/m9-vowel-transition.json
```
Expected: structured JSON with the report fields from Task 1.

**Step 5: Commit**

```bash
git add scripts/m9_vowel_transition_smoke.py packages/m9_sound/vowel_transitions.py tests/unit/test_m9_vowel_transitions.py
git commit -m "feat(m9): add vowel transition smoke CLI"
```

---

### Task 3: Wire documentation and project status to the new milestone

**Objective:** Promote the new workflow into the project docs so the next contributor sees the real next step.

**Files:**
- Modify: `docs/m9_status.md`
- Modify: `docs/m9_temporal_training_workflow.md` if the transition work becomes the new downstream note, or create `docs/m9_vowel_transition_workflow.md`
- Optional modify: `README.md` if it has a project-status section

**Step 1: Update the docs after the code is green**

Add a short note that says:
- temporal pure-tone work is validated
- vowel-transition family coverage is the next milestone
- the new smoke harness is the entry point
- the spectral substrate checkpoints are the runtime source of truth

**Step 2: Re-run the relevant tests and smoke**

Run:
```bash
.venv/bin/python -m pytest tests/unit/test_m9_vowel_transitions.py -q
.venv/bin/python scripts/m9_vowel_transition_smoke.py --family default --output /tmp/m9-vowel-transition.json
```

**Step 3: Commit**

```bash
git add docs/m9_status.md docs/m9_vowel_transition_workflow.md README.md
git commit -m "docs(m9): promote vowel transition coverage workflow"
```

---

## First Slice Recommendation

Do not start by broadening the whole vowel space.

Build the smallest useful vertical slice first:
1. one deterministic 3-step vowel-transition family
2. one held-out family
3. one noisy-input regression
4. one CLI smoke that emits JSON and a WAV artifact

If that slice is stable, then widen the family set.

## Verification Gates

Before calling the work done:
- focused tests pass
- CLI smoke passes
- JSON output is stable and machine-readable
- WAV export is valid with stdlib `wave`
- docs reflect the actual current next step
- the repo status is updated only after the new behavior is verified

## Notes / Constraints

- Keep the helper reusable; do not bury logic only in the CLI.
- Keep the CLI thin.
- Preserve deterministic seeds for repeatability.
- Prefer the existing spectral checkpoints; do not invent new artifact formats unless required.
- Keep the pure-tone temporal trainer intact; this is a new coverage layer, not a replacement.
