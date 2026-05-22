# M4 Live Adapter Fallback Fix

**Date:** 2026-05-21

## What changed

The live streaming runtime now falls back to the resolved attribution confidence when a clean WORLD frame has no explicit M4 confidence attached.

## Why this was needed

Before the fix, live WORLD frames could resolve a speaker ID but still skip adaptive-threshold learning because `m4_confidence` was `None`.

The result was:
- speaker identification worked
- WORLD frames were clearly present
- but the adaptive-threshold adapter never learned a non-self profile

## Fix

In `packages/pfc/streaming_runtime.py`:
- `_update_adapter_from_clean_world_evidence(...)` now uses `attr_result.confidence` as a fallback when `m4_confidence` is missing

## Regression test

Added a regression case to:
- `tests/integration/test_synthetic_adaptive_threshold.py`

The test verifies that a clean WORLD attribution with missing `m4_confidence` still updates the speaker adapter.

## Verification

Ran:
- `pytest tests/integration/test_synthetic_adaptive_threshold.py -q`
- `scripts/m4_live_adaptive_threshold.py` with pre-registered refs + world-audio playback

Result:
- adapter summary now includes `negative_live`
- live validation passes all criteria

## Outcome

This is the smaller fix, and it works. The live stack now learns from WORLD evidence instead of silently skipping those frames.
