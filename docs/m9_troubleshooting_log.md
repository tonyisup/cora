# M9 Acoustic Imitation — Troubleshooting Log

## Session: 2026-05-27 to 2026-05-29

### Goal
Build a multi-frequency acoustic imitation system that can produce different sounds matching different stimuli, on real audio hardware.

### Key Iterations

#### 1. Playback Trigger Bug
**Symptom:** No audio played when running `m9_live_test.py`
**Root cause:** Exact equality check `pos == lead_in_samples + n` almost never matched due to 1024-sample block alignment
**Fix:** Changed to boolean flag pattern: `if not playback_started and pos >= lead_in_samples:`
**Lesson:** Always use flag patterns for one-shot triggers in streaming loops

#### 2. Domain Gap: Synthetic vs Live Audio
**Symptom:** Controller trained on synthetic audio produced garbage on live captures
**Root cause:** Substrate hidden states for clean sine waves ≠ hidden states for noisy mic captures with room noise, distortion, and frequency response variations
**Evidence:** Hidden state distance between synthetic and live = 8.5 (should be ~0 for same input)
**Fix:** 1) Train substrate on live captured audio (not synthetic), 2) high-pass filter (100 Hz) to remove room/HVAC noise
**Lesson:** Training and deployment must use the same input distribution

#### 3. Laptop Fan Noise
**Symptom:** All demos sounded muddled regardless of processing
**Root cause:** Loud laptop fan next to mic was picked up as significant low-frequency energy
**Evidence:** STFT showed dominant energy at 8-55 Hz (fan fundamental + harmonics) across ALL frames
**Fix:** Turned off laptop fan
**Lesson:** Room noise matters — always check for environmental noise sources

#### 4. Phase Discontinuity at Frame Boundaries
**Symptom:** Audible clicks/warble at every ~30ms frame boundary
**Root cause:** Each frame generated as `sin(2πft + φ)` with independent φ per frame
**Evidence:** Clear discontinuities in waveform at frame boundaries
**Fix:** Track cumulative phase — `cum_phase += 2π·freq·frame_duration`, each frame starts where previous ended
**Result:** Warble eliminated, smooth continuous output

#### 5. Frame-to-Frame Frequency Jitter
**Symptom:** "Garbled" sound — too many frequency changes per second
**Root cause:** Controller outputs independent params every 30ms; rapid jumps (207→222→218→207 Hz) create audible modulation
**Fix:** Exponential moving average (EMA) smoothing: `freq_smoothed = α·freq_new + (1-α)·freq_prev` with α=0.3
**Result:** Frequency std reduced from ~109 Hz to ~55-92 Hz

#### 6. Controller Overfitting
**Symptom:** Large controller (2632 params) produced erratic output on live data
**Root cause:** Too much capacity for the training data (105 live captures); memorized noise instead of learning pattern
**Fix:** Smaller controller (336 params, 8 hidden units) + longer training (1000 epochs)
**Result:** Within-segment frequency std ~7 Hz (vs ~80 Hz before)

#### 7. Real-Time Feedback Loop Instability
**Symptom:** Running controller frame-by-frame during playback created oscillation
**Root cause:** Controller output fed back through substrate, which the controller then read — unstable feedback loop since controller wasn't trained on its own output distribution
**Fix:** Pre-compute all controller params from stimulus-encoded substrate states, then play back without feedback
**Result:** Clean, predictable output that's reproducible

### Final Architecture

```
Stimulus (tones) → [Filter: 100Hz HPF] → Mic capture
    → Substrate (RNN, live-trained) → Hidden state sequence
    → Controller (2-layer MLP, 336 params) → Per-frame oscillator params
    → EMA smoothing (α=0.3) → Smoothed params
    → Phase-continuous oscillator → Imitation audio
```

### Results

| Metric | Initial | After Fixes |
|--------|---------|-------------|
| Warble/clicks | Severe | None |
| Frequency jitter (std) | ~109 Hz | ~7 Hz per segment |
| Distinct tones audible | No | Yes (2-3 clear tones) |
| Temporal pattern | Garbled | Clear transitions |
| Live hardware score | N/A | 0.87-0.94 |

### Key Files
- `scripts/m9_temporal_smooth_demo.py` — Current best demo
- `data/m9_controller_temporal_v3.ckpt.json` — Best controller (336 params)
- `data/m9_substrate_live.ckpt.json` — Live-trained substrate
- `packages/m9_sound/producer.py` — Phase-continuous frame generation
