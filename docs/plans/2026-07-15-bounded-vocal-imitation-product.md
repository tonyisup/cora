# Bounded Vocal-Imitation Product Direction Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make the validated, bounded project-owner vocal-imitation loop Cora-Infans’ primary product and engineering direction, without reopening falsified Stage 2 or H4 claims.

**Architecture:** Treat the frozen V4 result as retained evidence, not a reusable training set. Add a separately specified product-session layer that orchestrates authorized capture, a fixed finite prototype bank, frozen-before-evaluation association, deterministic artifacts, and a user-facing replay/demo report. The product reports imitation fidelity separately from self-model fidelity and never presents a finite association result as continuous perception-production closure.

**Tech Stack:** Python 3.13+, existing `cora_infans` CLI, XVF3800 16 kHz/two-fused-channel capture path, deterministic WAV/JSON/HTML artifacts, pytest, NumPy.

---

## Non-negotiable boundaries

- **Primary product claim:** bounded, target-specific vocal imitation for an authorized project owner over a fixed finite repertoire.
- **Not claimed:** phoneme recognition, arbitrary speech imitation, speaker generalization, semantic understanding, child learning, continuous inverse closure, or new H4 evidence.
- **V4 evidence is immutable:** no code path may retrain, rescore, or overwrite the retained V4 real result.
- **No protected corpus access:** VocalSet, CMU Arctic protected/reserve speakers, and Stage 2 protected fieldsites remain inaccessible.
- **No automatic live I/O:** all microphone/speaker use stays explicitly authorized and fail-loud.
- **No raw audio committed:** raw recordings and generated audio stay outside Git; reports retain only manifests, hashes, metrics, and provenance.

## Product success definition

A product session is successful only when it can show all of the following:

1. an authorized source vocal target was captured through the declared microphone path;
2. the system selected or recalled a response only from a fixed, disclosed finite repertoire;
3. development associations froze before any held evaluation capture was scored;
4. each trial retains provenance for target capture, emitted response, response capture, and report;
5. the report distinguishes:
   - **imitation fidelity:** captured external target ↔ captured agent response;
   - **self-model fidelity:** predicted agent microphone capture ↔ actual captured agent response;
6. a product-facing demo clearly labels synthetic/replay fixtures versus real hardware evidence;
7. every gate failure produces an honest blocked/failed verdict, never a silent fallback or qualitative success claim.

---

### Task 1: Approve and document the product boundary

**Objective:** Make bounded vocal imitation the authoritative primary direction while retaining Stage 2/H4 stop rules.

**Files:**
- Create: `docs/BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md`
- Modify: `VISION.md`
- Modify: `README.md`
- Modify: `docs/PROJECT_STATUS_2026-07-12.md`

**Step 1: Write the product specification**

Document:

- target user and authorized-use boundary;
- the exact narrow product claim above;
- user-visible flow: prepare → authorized capture → finite association/recall → held evaluation → replayable report;
- forbidden claims and corpus boundary;
- source-of-truth artifacts and retention policy;
- separate imitation and self-model score definitions;
- stop/hold conditions;
- distinction between the historical V4 proof and future product sessions.

**Step 2: Amend `VISION.md`**

Add a short primary-direction section immediately after the Stage 5 constraint. It must authorize only this product protocol and link to the new product specification. It must explicitly say the direction does not reopen Stage 2 or H4.

**Step 3: Reframe the README front door**

Put the bounded vocal-imitation loop before synthetic plant infrastructure in the project narrative. Link the user-facing hero demo and product specification, and state the command remains blocked from live I/O unless explicitly authorized.

**Step 4: Update the dated status surface**

Record the strategic decision separately from the frozen scientific result. Do not alter V4 metrics or historical dispositions.

**Step 5: Verify documentation coherence**

Run:

```bash
git grep -n -i 'primary product\|continuous.*closure\|Stage 2\|H4' -- README.md VISION.md docs
```

Expected: product direction is present; stopped-program language remains intact; no document calls finite association H4 closure.

**Step 6: Commit**

```bash
git add VISION.md README.md docs/BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md docs/PROJECT_STATUS_2026-07-12.md
git commit -m "docs: set bounded vocal imitation as product direction"
```

---

### Task 2: Define a versioned product-session contract before writing runtime code

**Objective:** Freeze the operational data model and metrics for future product sessions without changing V4 evidence.

**Files:**
- Create: `docs/evidence/2026-07-15-bounded-vocal-imitation-product-session-preregistration.json`
- Create: `src/cora_infans/product_session.py`
- Create: `tests/test_product_session.py`

**Step 1: Write failing contract tests**

Add tests that require a product session manifest to:

```python
assert report["evidence_class"] in {"synthetic_fixture", "real_authorized"}
assert report["v4_evidence_mutated"] is False
assert set(report["artifacts"]) >= {
    "source_human_wav",
    "target_captured_wav",
    "agent_emitted_wav",
    "response_captured_wav",
    "self_prediction_wav",
    "report_json",
}
assert set(report["scores"]) >= {"imitation_fidelity", "self_model_fidelity"}
assert report["association_frozen_before_evaluation"] is True
```

Add a test that a report cannot claim `real_authorized` without an explicit authorization record and physical operating-point metadata.

**Step 2: Run the focused test to verify RED**

Run:

```bash
uv run --python 3.13 --with pytest --with numpy --with pydantic --with pyusb --with pyyaml --with libusb-package --with sounddevice pytest tests/test_product_session.py -q
```

Expected: FAIL because `product_session` does not exist.

**Step 3: Implement only pure report/manifest validation**

`src/cora_infans/product_session.py` must initially contain no live I/O. It should provide:

- immutable artifact-name schema;
- evidence-class validation;
- score separation validation;
- association-freeze validation;
- deterministic `report.json` construction;
- refusal messages for missing authorization or invalid provenance.

Do not add training, inference, or capture logic in this task.

**Step 4: Run focused tests to verify GREEN**

Run the Task 2 command again.

Expected: all `test_product_session.py` tests pass.

**Step 5: Commit**

```bash
git add src/cora_infans/product_session.py tests/test_product_session.py docs/evidence/2026-07-15-bounded-vocal-imitation-product-session-preregistration.json
git commit -m "feat: add bounded vocal imitation session contract"
```

---

### Task 3: Build an offline/replay product-session scorer

**Objective:** Establish reproducible measurement before any new real capture or controller work.

**Files:**
- Modify: `src/cora_infans/product_session.py`
- Create: `scripts/run_bounded_vocal_imitation_replay.py`
- Modify: `tests/test_product_session.py`
- Modify: `src/cora_infans/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Add failing tests for the canonical sensory target**

Tests must reject a trial that uses internal generated PCM as its imitation target when a microphone-captured external target is declared. Tests must accept a fully synthetic fixture only when it is labeled `synthetic_fixture`.

**Step 2: Verify RED**

Run:

```bash
uv run --python 3.13 --with pytest --with numpy --with pydantic --with pyusb --with pyyaml --with libusb-package --with sounddevice pytest tests/test_product_session.py tests/test_cli.py -q
```

Expected: failure on the new target-path and CLI contract tests.

**Step 3: Implement deterministic scores**

Implement non-learning, deterministic metrics for both score families:

- onset/offset alignment;
- duration delta;
- RMS-envelope distance;
- pitch-contour distance when voiced;
- mel/cepstral distance;
- an explicit unavailable/invalid score state for insufficient voicing.

Raw waveform MSE must not become the primary score. Save metric version and per-component values in `report.json`.

**Step 4: Add a replay-only CLI**

Expose a command such as:

```bash
python -m cora_infans --score-vocal-imitation-session --session-dir /path/to/session
```

It reads existing artifacts only, writes a deterministic report, and has no audio-device imports or side effects.

**Step 5: Verify GREEN and structural artifacts**

Run the focused tests, then run the replay CLI against a synthetic test fixture. Verify `report.json` contains both score families and all artifact hashes.

**Step 6: Commit**

```bash
git add src/cora_infans/product_session.py src/cora_infans/cli.py scripts/run_bounded_vocal_imitation_replay.py tests/test_product_session.py tests/test_cli.py
git commit -m "feat: score bounded imitation sessions offline"
```

---

### Task 4: Add a product-facing replay/demo surface

**Objective:** Make the bounded result understandable without overstating it.

**Files:**
- Modify: `src/cora_infans/hero_demo.py`
- Modify: `src/cora_infans/cli.py`
- Modify: `tests/test_hero_demo.py`
- Modify: `README.md`

**Step 1: Write failing hero-report tests**

Require the hero report and HTML to display:

- evidence class (`synthetic_fixture`, retained real V4 evidence, or future authorized session);
- frozen finite repertoire size;
- no-generalization disclaimer;
- imitation score and self-model score as separate sections;
- source/target/response provenance links or hash references;
- explicit “not H4 closure” language.

**Step 2: Verify RED**

Run:

```bash
uv run --python 3.13 --with pytest --with numpy --with pydantic --with pyusb --with pyyaml --with libusb-package --with sounddevice pytest tests/test_hero_demo.py -q
```

Expected: failure on the new presentation contract.

**Step 3: Implement deterministic presentation only**

Extend the existing hero demo so it renders retained V4 summary metrics and replay-session reports. Do not make the hero demo itself capture audio or train a new association.

**Step 4: Verify GREEN**

Run the focused hero test and generate an offline hero artifact:

```bash
python -m cora_infans --hero-demo --source synthetic --out-dir /private/tmp/cora-bounded-imitation-hero
```

Inspect `hero-demo.json` for the required labels and `hero-demo.html` for the claim boundary.

**Step 5: Commit**

```bash
git add src/cora_infans/hero_demo.py src/cora_infans/cli.py tests/test_hero_demo.py README.md
git commit -m "feat: present bounded vocal imitation evidence"
```

---

### Task 5: Specify—not yet execute—the next authorized real product-session protocol

**Objective:** Design the next real session without treating it as a continuation of V4 or consuming protected data.

**Files:**
- Create: `docs/BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md`
- Create: `docs/evidence/2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json`
- Modify: `docs/BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md`

**Step 1: Define the collection and retention decision**

Specify explicit owner consent, retention location, deletion behavior, device route, capture duration, take counts, and the exact authorization confirmation required before any microphone/speaker call.

**Step 2: Freeze evaluation before capture**

Specify development versus held takes, association-freeze point, required artifact names, score thresholds, no-retry rules, and what happens after a failure.

**Step 3: Freeze product repeatability requirements**

A first product-level claim should require at least:

- multiple fresh sessions, each with a fresh manifest;
- per-vowel and aggregate scores;
- explicit failure rows;
- report of both improvement from rest and held-evaluation fidelity;
- no changes to representation, repertoire, or thresholds after the first session’s results.

The exact repeat count and score thresholds must be decided before capture; do not borrow V4’s scientific thresholds automatically.

**Step 4: Review before implementation or hardware use**

No real capture command is added or run in this task. Present the protocol for explicit approval because it crosses the hardware and personal-recording boundary.

**Step 5: Commit**

```bash
git add docs/BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md docs/BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md docs/evidence/2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json
git commit -m "docs: specify bounded vocal imitation product session"
```

---

### Task 6: Final verification and status synchronization

**Objective:** Prove the product session surfaces are coherent before making a new product claim.

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_STATUS_2026-07-12.md` or a new current-dated status note

**Step 1: Run focused product tests**

```bash
uv run --python 3.13 --with pytest --with numpy --with pydantic --with pyusb --with pyyaml --with libusb-package --with sounddevice pytest tests/test_product_session.py tests/test_hero_demo.py tests/test_vocal_imitation.py -q
```

**Step 2: Run the full hardware-safe suite**

```bash
uv run --python 3.13 --with pytest --with numpy --with pydantic --with pyusb --with pyyaml --with libusb-package --with sounddevice pytest tests -q
```

**Step 3: Run readiness**

```bash
python -m cora_infans --operational-readiness
```

**Step 4: Update status only from executed output**

Document the actual test count, artifact paths, and claim boundary. Do not add a real-session success claim unless a separately approved real protocol has passed.

**Step 5: Final commit**

```bash
git add README.md docs/PROJECT_STATUS_2026-07-12.md
git commit -m "docs: report bounded imitation product verification"
```

---

## Acceptance checklist

- [ ] `VISION.md` names bounded vocal imitation as the primary product direction.
- [ ] Stage 2 and H4 remain stopped and protected-data boundaries remain intact.
- [ ] V4 remains immutable retained evidence.
- [ ] Product reports require the five core artifacts and provenance hashes.
- [ ] Imitation fidelity and self-model fidelity are never collapsed into one score.
- [ ] Offline/replay scoring has no device side effects.
- [ ] Hero output exposes evidence class and claim boundaries visibly.
- [ ] A future real-session protocol is approved before any new owner recording.
- [ ] Focused tests, full hardware-safe suite, and operational readiness all pass.
