# M9 Sound-First Motor Learning — Status & Plan

> Status: Multi-frequency temporal imitation working on hardware. Next: richer sounds.

## What Works

### Single Frequency Imitation
- 220-660 Hz range, all scores >0.91 on live hardware
- Clean, sustained tones with minimal jitter
- Latency: <100ms (capture → encode → produce)

### Multi-Frequency Sequences
- 2-tone patterns (e.g., 220→440 Hz): scores 0.88-0.94
- 3-tone patterns: scores 0.84-0.89
- Clear audible pitch transitions between segments

### Architecture
- **Substrate**: RNN (32 hidden units), pre-trained on live audio for frequency classification
- **Controller**: 2-layer MLP (336 params), trained with closed-loop feedback
- **Oscillator**: Phase-continuous waveform generation
- **Post-processing**: EMA smoothing (α=0.3), noise filtering (100 Hz HPF)

### Key Files
- `scripts/m9_temporal_smooth_demo.py` — Live demo with phase continuity + smoothing
- `data/m9_controller_temporal_v3.ckpt.json` — Best controller
- `data/m9_substrate_live.ckpt.json` — Live-trained substrate
- `docs/m9_troubleshooting_log.md` — Detailed iteration history

### Key Bugs Fixed
1. Playback trigger condition (exact equality → flag)
2. Phase discontinuity at frame boundaries (cumulative phase tracking)
3. Room noise contamination (high-pass filter)
4. Controller overfitting (smaller architecture)
5. Real-time feedback instability (pre-compute params)

## Next: Richer Sounds

### Current Limitation
Pure sine waves sound artificial. Speech requires **formants** — resonant frequency bands that give vowels their character.

### Approach: Formant Synthesis

Instead of `sin(2πft)`, use a **formant oscillator** that produces multiple simultaneous resonances:

```
output = Σᵢ Aᵢ · sin(2π·Fᵢ·t + φᵢ)
```

Where F₁, F₂, F₃ are formant frequencies (e.g., 700, 1200, 2500 Hz for vowel /ɑ/).

### Controller Changes
- Output: [F₁, F₂, F₃, A₁, A₂, A₃, waveform] instead of just [freq, amp, phase]
- Still 2-layer MLP, but output dim increases from 4 to 7
- Training: use formant-rich stimuli (vowel-like sounds) instead of pure tones

### Stimulus Generation
- Use Praat or similar to generate synthetic vowels
- Or record human vowel sounds (with permission)
- Target: 6-8 vowel-like sounds covering the vowel space

### Success Metric
- Can a listener distinguish between 4+ different "vowel" imitations?
- Formant frequency error < 50 Hz for each formant

### Timeline
1. Generate formant training stimuli (1-2 days)
2. Retrain live substrate for formant classification (1 day)
3. Train formant controller (1 day)
4. Validate on hardware (1 day)
