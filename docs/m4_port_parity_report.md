# M4 Port Parity Report

**Date:** 2026-05-22

## Summary

The M4 self-voice / arbitration port from `/Users/juicebox/Projects/corollary` has been brought into parity in `/Users/juicebox/Projects/cora` for the runtime, detector plumbing, live harnesses, validation scripts, and targeted regression coverage.

## What was ported

### Runtime and detector plumbing
- `packages/stg/speaker.py`
- `packages/pfc/streaming_runtime.py`

This includes:
- prototype-bank detector support
- gated bank+scorer arbitration
- `m4_mode` selection plumbing
- bank/scorer path selection in the streaming runtime

### Live scripts
- `scripts/m4_live_adaptive_threshold.py`
- `scripts/m4_mixed_live_embeddings.py`
- `scripts/m4_live_finetuned_test.py`
- `scripts/m4_validate_live_scorer.py`
- `scripts/m4_quick_negative_test.py`
- `scripts/m4_replay_attribution.py`
- `scripts/m4_capture_negatives_current_room.py`
- `scripts/m4_build_prototype_bank.py`
- `scripts/m4_validate_prototype_bank.py`

### Regression coverage
- `tests/unit/test_m4_gated_arbitration.py`
- `tests/unit/test_m4_mode_selection.py`
- `tests/unit/test_m4_plumbing.py`

## Verification

Ran:
- `python3 -m py_compile` on the edited M4 scripts
- `uv run --with pytest --with numpy --with torch pytest tests/unit/test_m4_plumbing.py tests/unit/test_m4_gated_arbitration.py tests/unit/test_m4_mode_selection.py -q`

Result:
- `9 passed`

## Notes

- `uv.lock` was removed after test runs to avoid lockfile churn.
- The parity pass confirmed the remaining M4 live-script diffs were eliminated.
- The only outstanding work is broader validation outside the directly ported M4 surface, if desired.

## Next steps

1. Run a live hardware smoke test using one of the updated M4 harnesses.
2. If the hardware path is stable, promote the best M4 mode into the default operational path for the relevant runtime.
3. Add a short acceptance note to the relevant milestone doc if this port is being used as a release gate.
