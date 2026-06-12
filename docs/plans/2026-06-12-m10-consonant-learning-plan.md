# M10 Consonant Learning Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Teach M9 to detect and imitate consonants as first-class speech events, starting with consonant gating and a small live demo that can wait for speech, separate consonants from vowels, and replay a short consonant-bearing phrase.

**Architecture:**
Build M10 on top of the existing M9 vowel-transition stack instead of replacing it. Keep the current vowel demo and CE baseline as the regression anchor, then add a consonant-aware decoding path that can label frames as vowel / consonant / silence. Start with a tiny controlled consonant set and a live capture demo, then widen to stop/fricative/nasal coverage once the first slice is stable. Keep artifacts listenable and deterministic: JSON reports, WAV exports, and dry-run smoke tests.

**Tech Stack:** Python, NumPy, stdlib `wave`, existing `m9_sound` package, pytest, existing live-capture harnesses, existing vowel-transition demo scripts.

---

### Task 1: Define the consonant-first M10 slice and its acceptance gates

**Objective:** Document the smallest useful consonant milestone so implementation stays scoped and testable.

**Files:**
- Create: `docs/m10_consonant_plan.md` or keep this plan as the working spec
- Modify: `docs/m9_status.md`
- Modify: `docs/m10_goal.md` if the wording needs to be tightened after implementation starts

**Step 1: Write the acceptance criteria before coding**

Document the M10 first slice as:
- detect consonant activity in live mic input
- distinguish consonant frames from vowel frames and silence
- keep the existing vowel demo working unchanged
- produce a short listenable consonant demo from real or dry-run input

**Step 2: Write the regression boundaries**

Define what M10 is not yet trying to do:
- full phoneme coverage
- perfect accent normalization
- general speech cloning
- large-scale corpus training before the small consonant loop works

**Step 3: Update the project status note**

After the code is green, update `docs/m9_status.md` to say M9 is complete and M10 is consonants.

**Verification:**
- The M10 acceptance criteria are visible in docs before the first code change
- The docs clearly preserve the vowel system as a baseline

---

### Task 2: Add a consonant-aware decode path to the existing frame decoder

**Objective:** Extend the current vowel decoder so it can emit consonant or silence instead of forcing every non-silence frame into a vowel.

**Files:**
- Modify: `packages/m9_sound/vowel_transitions.py`
- Modify: `packages/m9_sound/live_capture.py`
- Test: `tests/unit/test_m10_consonant_decode.py`

**Step 1: Write failing tests**

Add tests that assert:
- a frame far from all vowel centroids can be labeled as consonant/silence
- vowel frames still decode to the original vowel labels
- silence frames remain silence
- the decode report can include a consonant count or consonant rate

Example test shape:
```python
def test_consonant_gate_rejects_far_frames():
    report = decode_vowel_transition_audio(
        audio,
        consonant_gate=2.0,
        silence_threshold=0.0,
    )
    assert "_" in report["decoded_vowels"]
```

**Step 2: Run the tests and verify failure**

Run:
```bash
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_decode.py -q
```
Expected: fail until the decode path explicitly supports consonant-aware behavior.

**Step 3: Implement the minimal decoder change**

In `packages/m9_sound/vowel_transitions.py`:
- keep the existing centroid-based vowel decode
- add a consonant gate threshold
- when the nearest-centroid distance exceeds the threshold, emit `_` instead of a vowel
- include consonant-related counts in the returned report

In `packages/m9_sound/live_capture.py`:
- thread the new gate parameter through the live capture report functions
- keep the current vowel report compatible so existing scripts do not break

**Step 4: Run the tests to verify pass**

Run:
```bash
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_decode.py -q
```
Expected: pass.

**Step 5: Commit**

```bash
git add packages/m9_sound/vowel_transitions.py packages/m9_sound/live_capture.py tests/unit/test_m10_consonant_decode.py
git commit -m "feat(m10): add consonant-aware decode gate"
```

---

### Task 3: Add a small consonant capture set and a listenable demo script

**Objective:** Create a minimal live demo that can wait for the user to speak, capture consonant-bearing utterances, and emit listenable WAV artifacts.

**Files:**
- Create: `scripts/m10_live_consonant_demo.py`
- Create: `scripts/m10_consonant_smoke.py`
- Modify: `scripts/m9_live_voice_imitate_demo.py` only if you want it to share the new gate plumbing
- Test: `tests/unit/test_m10_consonant_demo.py`

**Step 1: Write failing tests**

Add tests that assert:
- the demo script parses the consonant-specific flags
- dry-run mode produces a JSON report and WAV artifacts
- the report records the detected consonant/silence split
- the existing vowel demo still runs unchanged

Example command expectation:
```bash
./.venv/bin/python scripts/m10_live_consonant_demo.py --dry-run --once --output-dir /tmp/m10-demo
```
Expected: JSON report plus rendered/captured WAV files.

**Step 2: Run the tests and verify failure**

Run:
```bash
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_demo.py -q
```
Expected: fail until the new script exists.

**Step 3: Implement the demo thinly**

Keep the script simple:
- wait/listen for mic speech
- classify consonant-bearing frames
- route through the new consonant gate
- save rendered/captured WAVs and JSON
- support `--dry-run` for hardware-free regression testing

Suggested flags:
- `--families`
- `--listen-family`
- `--consonant-gate`
- `--silence-threshold`
- `--threshold`
- `--output-dir`
- `--iterations`
- `--dry-run`

**Step 4: Run the demo smoke**

Run:
```bash
./.venv/bin/python scripts/m10_consonant_smoke.py --dry-run --output /tmp/m10-consonant-smoke.json
```
Expected: valid JSON report and valid WAV metadata.

**Step 5: Commit**

```bash
git add scripts/m10_live_consonant_demo.py scripts/m10_consonant_smoke.py tests/unit/test_m10_consonant_demo.py
git commit -m "feat(m10): add consonant demo and smoke harness"
```

---

### Task 4: Build a tiny consonant corpus and label it consistently

**Objective:** Collect a small controlled dataset that gives M10 real consonant examples instead of only vowel-adjacent artifacts.

**Files:**
- Create: `scripts/m10_record_consonants.py`
- Create: `data/m10_consonant_prompts.json` or a similar prompt manifest
- Test: `tests/unit/test_m10_consonant_prompts.py`

**Step 1: Write failing tests**

Add tests that assert:
- the prompt manifest contains the planned consonant inventory
- each prompt maps to a stable label
- the manifest can be loaded by the recording script

Example labels:
- stops: `p`, `t`, `k`
- fricatives: `s`, `f`, `sh`
- nasals: `m`, `n`

**Step 2: Run the tests and verify failure**

Run:
```bash
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_prompts.py -q
```
Expected: fail until the prompt manifest exists.

**Step 3: Implement the smallest corpus tool**

The recording script should:
- prompt the user for a small set of consonant-bearing syllables or CVC tokens
- save WAVs with stable filenames
- write a JSON manifest with speaker, token, label, and file path
- remain explicit about microphone/device settings

**Step 4: Verify the artifacts**

Run the recording script with a dry-run or short controlled set, then confirm:
- WAVs exist
- manifest JSON is readable
- labels are consistent and reproducible

**Step 5: Commit**

```bash
git add scripts/m10_record_consonants.py data/m10_consonant_prompts.json tests/unit/test_m10_consonant_prompts.py
git commit -m "feat(m10): add initial consonant corpus capture"
```

---

### Task 5: Wire M10 into the docs and keep M9 as the regression baseline

**Objective:** Make the milestone transition obvious so future work starts in the right place.

**Files:**
- Modify: `docs/m9_status.md`
- Modify: `docs/m10_goal.md`
- Create or modify: `docs/plans/2026-06-12-m10-consonant-learning-plan.md`
- Optional modify: `README.md` if it has a visible milestone summary

**Step 1: Update the docs after code passes**

Add a short note that says:
- M9 is complete
- vowel imitation remains the baseline regression suite
- M10 is consonants
- the new consonant demo is the entry point for the next phase

**Step 2: Re-run the relevant smoke tests**

Run:
```bash
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_decode.py -q
./.venv/bin/python -m pytest tests/unit/test_m10_consonant_demo.py -q
./.venv/bin/python scripts/m9_live_voice_imitate_demo.py --dry-run --once
```
Expected: all pass, and the existing vowel demo still behaves exactly as before.

**Step 3: Commit**

```bash
git add docs/m9_status.md docs/m10_goal.md docs/plans/2026-06-12-m10-consonant-learning-plan.md README.md
git commit -m "docs(m10): define consonant learning milestone"
```

---

## First Slice Recommendation

Do not start with full phoneme modeling.

Build the smallest useful vertical slice first:
1. consonant gate in the existing decoder
2. one dry-run consonant demo
3. one tiny labeled consonant corpus
4. one regression test that proves consonants are not forced into vowels

If that slice is stable, then broaden the consonant inventory.

## Verification Gates

Before calling M10 started:
- the consonant gate exists and is tested
- the live demo can run in dry-run mode
- WAV + JSON artifacts are valid
- M9 vowel demos still pass unchanged
- the docs say M9 is done and M10 is consonants

## Notes / Constraints

- Keep M9 intact as the regression baseline.
- Favor detection and gating before trying to synthesize perfect consonants.
- Keep the demo thin; the reusable logic belongs in `packages/m9_sound`.
- Do not let this turn into an open-ended speech-recognition project yet.
- Start with a small consonant set and a short phrase inventory.
