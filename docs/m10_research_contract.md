# M10 Research Contract

**Date:** 2026-06-14  
**Status:** Active 48k/full-inventory research loop  
**Reference Arbor run:** `.arbor/sessions/run_20260614_124107/REPORT.md`

## Objective

Teach the M10 path to support full consonant inventory while keeping the existing M9 vowel/temporal baseline stable.

The immediate research question is:

> Can 48k speaker→mic capture plus targeted rare-label collections make the full consonant inventory learnable without regressing vowel behavior or the M9 16k baseline?

## Current Baseline

### Previous 16k / core-consonant baseline

- Eval accuracy: 54.7% (248/453)
- Macro accuracy: 60.2%
- Hard-label avg (`k/sh/th/f/s`): 49.8%
- Weakest label: `k` at 43.8%
- Readout: `REVIEW`

### First 48k / full-inventory run

- Eval accuracy: 60.7% (167/275)
- Macro accuracy: 38.7%
- Hard-label avg (`k/sh/th/f/s`): 51.9%
- Weakest label: `b` at 0.0% (`n=1`)
- Readout: `REVIEW`

Interpretation:

- 48k improved overall eval and slightly improved the hard-label average.
- Macro is low because the full-inventory eval split has many one-shot labels.
- The model is learning some high-frequency consonants (`f`, `sh`, `s`, `n`) but rare labels are under-sampled.
- The bottleneck is now data coverage, class imbalance, and the vowel/consonant sample-rate bridge — not just articulation.

## Primary Metrics

Use these in order:

1. **Hard-label avg** for `k/sh/th/f/s`
   - Target: `>= 57%`
   - Current 48k: `51.9%`
2. **Macro accuracy over meaningful labels**
   - Only compare macro once each evaluated label has enough examples.
   - Suggested minimum: `n >= 10` per label.
3. **Overall eval accuracy**
   - Should improve or at least not regress.
4. **Per-label floor**
   - No important label with `n >= 10` should sit below roughly `45%`.
5. **M9/vowel regression guard**
   - Vowel behavior and smoke/live-demo baselines should not regress.

## B_dev / B_test Discipline

Borrowed from Arbor:

- **B_dev:** current `data/m10/manifests/captures.jsonl` evaluated by `scripts/m10_train_v3.py`.
- **B_test:** a later frozen collection session or held-out targeted batch that is not used for training.
- Do not use B_test for iteration.
- Do not change the eval harness to game the score.
- Code/data changes should only be kept if they improve the meaningful metric without regressing the guardrails.

For now, the current hash split is acceptable as B_dev, but macro is not trustworthy until rare labels have enough eval examples.

## Protected Paths

Do not modify these without a separate explicit decision:

- `packages/m9_sound/temporal.py`
- `data/m9_*`
- `scripts/m10_consonant_smoke.py`, except for metric plumbing
- `uv.lock`, except for intentional environment changes

Append-only / treat carefully:

- `data/m10/raw/**`
- `data/m10/manifests/captures.jsonl`

Allowed working area:

- `scripts/m10_record_human.py`
- `scripts/m10_train_v3.py`
- `scripts/m10_live_consonant_demo.py`
- M10 docs under `docs/`

## Arbor Lessons to Preserve

From the first Arbor run:

- Keep shared decoder/listen-window code untouched unless there is a strong reason.
- Localize demo policy changes before touching shared M9 infrastructure.
- Use a Probe Block before ideation.
- Use four-line hypotheses:
  - `Mechanism`
  - `Hypothesis`
  - `Observable`
  - `Conflicts`
- Prune shallow tweaks that do not attack a class-level bottleneck.
- Do not trust incomplete runs or metric summaries if the benchmark setup is unclear.

## Current Probe Block

```text
PROBE BLOCK
Q1 First principles : class coverage / representation bottleneck - evidence: 48k macro 38.7%, rare labels with n=1-4, hard-label avg only 51.9%
Q2 Hidden assumption: current one-shot rare-label examples are enough to learn full inventory - if dropped: targeted multi-context collections and class-balanced sampling become the next lever
Q3 Elephant         : vowel/consonant bridge is split across 16k M9 and 48k M10 paths without a clean sample-rate abstraction
Q4 Hamming          : yes - the current gap is close enough that data coverage, class balance, and bridge plumbing are plausible next fixes rather than a full model rewrite
```

## Next Commands

### Targeted rare-label collection

```bash
./.venv/bin/python scripts/m10_record_human.py \
  --speaker juicebox \
  --labels b,d,g,th,v,z,l,r,w,y \
  --reps 2 \
  --shuffle
```

If the first batch is easy, increase to `--reps 3` or run a second targeted batch.

### 48k training after targeted collection

Because the class set is unchanged, resume from the existing 48k checkpoint:

```bash
./.venv/bin/python scripts/m10_train_v3.py \
  --epochs 300 \
  --resume-from data/m10_v3.ckpt.json
```

Use `--resume-from none` only for major changes such as feature extraction, class set, architecture, or domain changes.

## Definition of Done

This research loop is done when:

- Targeted rare-label collections have enough eval examples to make macro meaningful.
- 48k hard-label avg reaches at least `57%`.
- Overall eval does not regress.
- No important label with sufficient eval examples falls below roughly `45%`.
- Vowel behavior and smoke/live-demo baselines remain stable.
- The vowel/consonant sample-rate bridge is explicit and documented.
