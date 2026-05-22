# M4 Live Validation Report

**Date:** 2026-05-21

## What I ran

I exercised the live hardware validator on the current M4 stack using:
- input device: `1` (`BL-2021USB`)
- output device: `3` (`Mac mini Speakers`)

Runs performed:
1. self-voice live validation
2. live validation with pre-registered speaker references
3. live validation with pre-registered references plus world-audio playback

## Results

### Run 1: self-voice only

- M4 attempted on live audio: pass
- M4 accepted self frames: pass
- speaker identification active: fail
- adaptive-threshold profile behavior: pass

This showed the live M4 path was active, but no speaker ID was being resolved yet.

### Run 2: pre-registered speaker references

- M4 attempted on live audio: pass
- M4 accepted self frames: pass
- speaker identification active: pass
- adaptive-threshold profile behavior: fail

This proved speaker matching was happening, but the adapter did not accumulate a non-self profile.

### Run 3: pre-registered references + world-audio playback

- M4 attempted on live audio: pass
- M4 accepted self frames: pass
- speaker identification active: pass
- adaptive-threshold profile behavior: fail

The WORLD path was active, but the live runtime did not emit M4 confidence for those WORLD frames, so the adapter had no confidence observations to update from.

## Ground truth from the logs

The live log showed:
- stable `speaker_best_match_id = negative_live`
- `speaker_match_passed = True` on 220 frames
- `speaker_id = negative_live` on the live-ID run
- `m4_confidence = None` for WORLD frames, which blocked adapter learning

## Interpretation

The live hardware path is operational:
- audio I/O works
- M4 fires on live audio
- speaker identification works
- world playback is captured and classified

The remaining gap is narrower than the validator initially suggested:
- the adaptive-threshold adapter is wired correctly
- but the current live path does not feed it WORLD-frame M4 confidences
- therefore non-self profile learning does not yet happen from live evidence

## Recommendation

For the current milestone state, the correct next step is a small runtime fix in the streaming pipeline:
- emit or retain M4 confidence for WORLD-labeled frames that should update the adapter
- or explicitly feed the adapter from a clean WORLD post-pass once the frame is resolved

That is a code-level integration fix, not a data problem.

## Files / artifacts

- `/tmp/m4_live_adaptive.json`
- `/tmp/m4_live_adaptive_registered.json`
- `/tmp/m4_live_adaptive_registered_025.json`
- `/tmp/m4_live_adaptive_world.json`
- `/tmp/m4_live_adaptive.jsonl`
- `/tmp/m4_live_adaptive_registered.jsonl`
- `/tmp/m4_live_adaptive_registered_025.jsonl`
- `/tmp/m4_live_adaptive_world.jsonl`
- `/tmp/m5_live_capture_live/`
- `/tmp/m5_speaker_refs/`

## Bottom line

Live validation happened and the stack is real, but the final adapter-learning piece is not yet live-clean. The hardware path is ready; the last step is a small runtime integration fix for WORLD-frame confidence propagation.
