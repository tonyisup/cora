# M10 Idea Tree

**Date:** 2026-06-14  
**Root goal:** full consonant inventory with a stable vowel/consonant bridge  
**Reference:** `docs/m10_research_contract.md`

## Baseline

- 16k / core-consonant baseline:
  - Eval: 54.7%
  - Macro: 60.2%
  - Hard-label avg (`k/sh/th/f/s`): 49.8%
  - Weakest: `k` 43.8%
- 48k / full-inventory first run:
  - Eval: 60.7%
  - Macro: 38.7%
  - Hard-label avg (`k/sh/th/f/s`): 51.9%
  - Weakest: `b` 0.0% (`n=1`)

## ROOT: Full inventory with stable vowel/consonant bridge [RUNNING]

**Insight:** 48k is helping overall eval and some high-frequency consonants, but full-inventory macro is not meaningful yet because rare labels are under-sampled. The next bottleneck is data coverage, class balance, and an explicit sample-rate bridge.

**What works:**
- 48k capture path is real and usable.
- Some consonants are already learning (`f`, `sh`, `s`, `n`).
- Vowels remain strong.
- Targeted collection mode now exists in `scripts/m10_record_human.py`.

**What does not work yet:**
- Rare labels have too few eval examples to judge fairly.
- Macro is being dragged down by one-shot classes.
- The live demo is now 48k-aware, but the vowel/consonant bridge is still implicit.
- Shared M9 decoder changes are risky and should stay pruned unless there is a clear win.

---

## 1: Data coverage for rare labels [RUNNING]

**Mechanism:** Targeted multi-context collections for `b,d,g,th,v,z,l,r,w,y` with repeated captures and shuffled prompts.  
**Hypothesis:** Rare labels fail because the current split has too few examples, not because they are impossible. More examples should raise macro and stabilize the full-inventory eval.  
**Observable:** Each rare label reaches at least `n >= 10` eval examples; macro improves materially without dropping hard-label avg.  
**Conflicts:** None — this attacks a data-coverage bottleneck, not the model architecture.

### 1.1: First targeted rare-label batch [RUNNING]

**Mechanism:** Run `scripts/m10_record_human.py` with `--labels b,d,g,th,v,z,l,r,w,y --reps 2 --shuffle`.  
**Hypothesis:** Multi-context targeted capture will give the rare labels enough signal to move from one-shot noise to learnable classes.  
**Observable:** After retraining, rare labels show non-zero per-label accuracy and macro improves.  
**Conflicts:** None.

---

## 2: Class-balanced sampling / batching [PENDING]

**Mechanism:** Add class-balanced or per-class-weighted batching for the full-inventory trainer.  
**Hypothesis:** The model is dominated by common labels and vowels; balancing the training signal should help rare consonants without hurting vowels.  
**Observable:** Macro improves while hard-label avg stays stable or improves.  
**Conflicts:** Must not overfit the dev split or distort the eval contract.

---

## 3: Sample-rate bridge between 16k vowels and 48k consonants [PENDING]

**Mechanism:** Make the handoff explicit with a small config/bridge layer and explicit `sr` / `frame_size` propagation.  
**Hypothesis:** The current vowel and consonant paths work, but the bridge is implicit and fragile. A clean bridge will make future integration safer.  
**Observable:** Live demo and training both use model config consistently; no silent 16k/48k assumptions remain.  
**Conflicts:** Do not change `packages/m9_sound/temporal.py` globally; keep the M9 16k baseline intact.

---

## 4: Core vs full-inventory training split [PENDING]

**Mechanism:** Support separate `core` and `full` class sets for training and evaluation.  
**Hypothesis:** The full-inventory set is too sparse for a single metric right now. A core set gives a stable operating point while full inventory matures.  
**Observable:** Core eval remains stable; full inventory can be tracked separately once data coverage improves.  
**Conflicts:** Must not hide full-inventory failures; core and full metrics should both be reported.

---

## 5: Live-demo candidate policy [VALIDATED, NOT MERGED]

**Mechanism:** In dry-run, bias imitation toward `cv_nasal` at `duration_scale=0.8` and use model-backed scoring.  
**Hypothesis:** The live demo should not trust the first synthesized family; a small candidate policy can improve consonant evidence without touching shared decoder code.  
**Observable:** Dry-run live-demo imitation consonant_rate improves while smoke stays at `1.0`.  
**Conflicts:** Arbor validated this, but the branch was not merged because the worktree lacked runtime assets.

### 5.1: Arbor branch `coordinator/n3-5-mechanism-dry-run-default-imitat-1488eb76` [VALIDATED]

**Result:** Smoke stayed at `1.0`; live dry-run imitation consonant_rate reached about `0.95-1.0`.  
**Status:** Keep as a reference; port only if useful for the demo path.

---

## 6: Model redesign if 48k plateaus [PENDING]

**Mechanism:** If targeted collections and balancing do not move the needle, revisit representation or architecture.  
**Hypothesis:** The current model may still be too shallow for full inventory even with better data.  
**Observable:** Hard-label avg remains below target after data coverage improves.  
**Conflicts:** Do not jump here prematurely; data coverage and class balance must be tested first.

---

## Pruned / Avoid

- Changing `packages/m9_sound/temporal.py` to 48k globally.
- Shared decoder/filterbank edits that regress smoke or live-demo baselines.
- More articulation-only iteration after repeated flat results.
- Treating macro as meaningful before rare labels have enough eval examples.
