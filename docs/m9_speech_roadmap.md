# M9 Speech Imitation Roadmap

> Status: Historical roadmap for the earlier tone-imitation phase. The current M9 vowel-imitation path is working and live-capture retrain is validated; temporal sequences remain the next major gap.

## Where We Are

### Working
- Acoustic imitation loop: play stimulus (device 3) → capture (device 1) → producer generates imitation → play (device 2) → capture → score → REINFORCE update
- Two-layer controller (664 params) learns multiple frequencies without forgetting
- Checkpoint persistence saves/loads weights as JSON
- Similarity metric v4: mel-scale binning + octave-tolerant autocorrelation pitch
- Softened square waveform for cleaner live capture

### Verified Results
- Single frequency (220 Hz, 50 trials, lr=0.05): reward 0.66 → 0.73
- Multi-frequency curriculum (3 freqs × 10 trials): final eval 0.72–0.75 across all freqs
- No catastrophic forgetting (vs single-layer: all freqs collapsed to 0.62–0.68)

## What Speech Requires (and what's missing)

### 1. Temporal Structure (NEXT)
Speech is a rapid sequence of sounds, not a single tone. Need:
- Stimulus: sequence of frequencies (e.g., 220 Hz → 330 Hz → 440 Hz)
- Producer: must change parameters over time, not just sustain one tone
- Substrate: must learn temporal patterns, not just static state

### 2. Richer Oscillator (AFTER TEMPORAL)
Pure tones (sine, triangle, square) don't sound like speech. Need:
- Formant synthesis: multiple resonant frequencies simultaneously
- Vocal tract model or formant filter bank
- Nasal, fricative, plosive primitives

### 3. Segmentation and Sequencing
Speech has clear boundaries between phonemes. Need:
- Onset/offset detection in the audio stream
- Substrate that learns to segment continuous audio
- Controller that produces discrete gestural targets

### 4. Much Richer Audio Features
Current: MFCC-like spectral + pitch distance. Need:
- Formant tracking (F1, F2, F3)
- Spectral envelope dynamics
- Temporal derivative features

### 5. Scale
Current: ~30–100 trials. Speech needs: thousands to millions of examples
- Need efficient batch training (not one trial at a time)
- Need diverse stimuli (not just pure tones)

### 6. Real-Time Processing
Current: ~3–4 seconds per trial. Speech needs: real-time or near-real-time
- Optimize play/capture pipeline
- Possibly move scoring to GPU

## Recommended Sequence

### Near-term recommendation
1. Add a small regression harness around live-capture retrain so the current vowel path stays stable under noisy-room/device variation.
2. Then move to temporal sequences (2–3 vowel/frequency transitions per utterance).

### Phase 1: Temporal Sequences (next step)
- Stimulus: 2–3 frequency sequence (e.g., 220 Hz 0.3s → 330 Hz 0.3s)
- Producer must learn to switch frequencies mid-utterance
- Substrate must track temporal context
- Success metric: can the producer match a 2-tone sequence?

### Phase 2: Formant Sounds
- Replace pure tones with formant-rich stimuli (vowel-like)
- May need richer oscillator (formant filter bank)
- Controller learns to produce formant patterns
- Success metric: can the producer match a simple vowel?

### Phase 3: Phoneme Sequences
- Stimulus: sequences of 2–3 phoneme-like sounds
- Substrate learns to segment andrepresent temporal structure
- Success metric: can the producer match a simple CVC pattern?

### Phase 4: Scaling
- Batch training infrastructure
- Diverse speech corpus
- Optimization for real-time processing

## Key Design Decisions

1. **Build the curriculum**: simple → complex, always
2. **Verify each phase** before adding complexity
3. **Keep the loop intact**: stimulus → capture → imitation → score → update is the core; everything else serves this
4. **Measure everything**: score trajectories, weight norms, frequency-specific performance

## Files

- `scripts/m9_acoustic_imitation.py` — main imitation loop script
- `packages/m9_sound/controller.py` — two-layer neural controller
- `packages/m9_sound/producer.py` — producer with checkpoint save/load
- `packages/m9_sound/similarity.py` — v4 similarity metric
- `packages/m9_sound/oscillator.py` — waveform generators
- `docs/m9_live_audio_gate.md` — hardware validation results
- `docs/m9_sound_first.md` — M9 spec and status
