# M4 Live Smoke Test Notes

**Date:** 2026-05-22

## Goal

Exercise the updated M4 live runtime on real devices and verify the port is usable end-to-end, not just unit-tested.

## Environment

Device discovery via `sounddevice` showed:
- input `1` = `BL-2021USB`
- output `3` = `Mac mini Speakers`

The following on-demand packages were needed to run the live scripts in this repo:
- `sounddevice`
- `edge-tts`
- `scipy`
- `webrtcvad`
- `pydantic`
- `speechbrain`
- `pytest`
- `numpy`
- `torch`

## Smoke test 1: live adaptive threshold

Command used:

```bash
uv run --with sounddevice --with edge-tts --with scipy --with webrtcvad --with pydantic --with speechbrain --with pytest --with numpy --with torch python scripts/m4_live_adaptive_threshold.py \
  --input-device 1 \
  --output-device 3 \
  --duration 3.0 \
  --text 'Smoke test: M4 live adaptive threshold.' \
  --m4-scorer /Users/juicebox/Projects/corollary/data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt
```

Result:
- `M4 attempted`: 76
- `M4 accepted`: 33
- `Speaker IDs seen`: none
- overall criteria: partial fail, because speaker identification was not seeded in this run

This verified the M4 path was active on live hardware.

## Smoke test 2: mixed live embeddings

Command used:

```bash
uv run --with sounddevice --with edge-tts --with scipy --with webrtcvad --with pydantic --with speechbrain --with pytest --with numpy --with torch python scripts/m4_mixed_live_embeddings.py \
  --input-device 1 \
  --output-device 3 \
  --world-output-device 3 \
  --duration 6.0 \
  --text 'Smoke test: M4 mixed live embeddings.' \
  --m4-scorer /Users/juicebox/Projects/corollary/data/m4_finetuned_ecapa_live_domain_20260517/fine_tuned/scorer.pt \
  --register-speaker-dir /Users/juicebox/Projects/corollary/data/m4_human_negative_bl2021 \
  --play-world-audio /Users/juicebox/Projects/corollary/data/m4_human_negative_bl2021/human_negative_speaker2.npy
```

Result:
- live-path embeddings were successfully loaded from the corollary fixture directory
- speaker IDs were active:
  - `human_negative_speaker2`
  - `human_negative_speaker4`
  - `human_negative_speaker5`
- verdict counts included `SELF`, `WORLD`, and `BOTH`
- M4 attempted and accepted frames were non-zero

This is the better end-to-end smoke test for the current port because it exercises speaker registration plus world playback.

## Observed caveat

The `m4_live_adaptive_threshold.py` harness still needs a valid speaker-ID seed to show adapter learning on live runs. Without that, the runtime can still validate M4 attempts, but the adaptive-threshold profile remains empty.

## Takeaway

The port is live-functional:
- the runtime runs on real audio devices
- the fine-tuned scorer path works
- the mixed live-embedding path can register speakers and resolve IDs

The remaining work is mostly operational polish:
- make the most useful smoke-test fixture easier to supply from `cora`
- decide which live harness should be the recommended default for validation
