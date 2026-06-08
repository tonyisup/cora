# M9 Temporal Training Workflow

This note promotes the validated M9 temporal-training workflow into a reusable reference.

## Goal

Train the temporal controller against a real local reward signal that reflects:
- transition count
- transition ordering / timing
- phase continuity at frame boundaries
- held-out family generalization

## Workflow

1. Start from the migrated producer-format checkpoint:
   - `data/m9_controller_temporal_v3_migrated.ckpt.json`
2. Run the family-pair sweep to measure the family landscape:
   - `scripts/m9_temporal_family_sweep.py`
3. Use the sweep summary to derive the default held-out blend weight.
   - The trainer now prefers `--heldout-weight-source sweep`
   - `auto` held-out weight resolves to the sweep mean when available
4. Train with local in-memory evaluation:
   - `scripts/m9_temporal_train.py`
5. Inspect the emitted JSON and `.progress.json` / `.rewards.jsonl` artifacts.

## Recommended Commands

Primary sweep:
```bash
.venv/bin/python scripts/m9_temporal_family_sweep.py \
  --families default diphthong wide_sweep \
  --output data/m9_temporal_family_pair_sweep.json
```

Training run with sweep-derived weighting:
```bash
.venv/bin/python scripts/m9_temporal_train.py \
  --episodes 40 \
  --heldout-family wide_sweep \
  --heldout-weight auto \
  --heldout-weight-source sweep \
  --output data/m9_controller_temporal_trained.ckpt.json
```

## What the sweep report contains

- ordered family-pair rewards
- decoded transition counts
- per-pair auto-derived held-out weights
- summary means for primary reward, held-out reward, and held-out weight
- best pair by blended reward

## Why this matters

The earlier trainer only had a single held-out calibration pair. That was useful but brittle. The sweep summary makes the held-out blend weight reflect the broader temporal-family structure, which is a better proxy for generalization.

## Files

- `scripts/m9_temporal_train.py`
- `scripts/m9_temporal_family_sweep.py`
- `tests/unit/test_m9_temporal.py`
- `data/m9_temporal_family_pair_sweep.json`
- `data/m9_controller_temporal_train2_sweepweight.ckpt.json`
