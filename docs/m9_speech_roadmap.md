# M9 Speech Imitation Roadmap

> Status: Historical roadmap for the earlier tone-imitation phase. The current M9 vowel-imitation path is working and live-capture retrain is validated; the next immediate step is vowel-transition family coverage over the spectral/formant substrate path, with longer temporal sequences as the follow-on gap.

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

### 1. Vowel-transition family coverage (DONE)
Speech is a rapid sequence of vowel-like transitions, not just isolated steady states. The current slice is now implemented as a deterministic helper + smoke CLI.
- Stimulus: sequence of vowel transitions (e.g., /i/ → /ɛ/ → /u/)
- Producer: changes parameters over time, not just sustain one vowel
- Substrate: learns transition patterns, not just static state

### 2. Richer Oscillator (AFTER TRANSITIONS)
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
1. Keep the current vowel path stable under noisy-room/device variation with a small live-capture retrain regression harness.
2. Then extend beyond the new long_arc and formant_rich slices toward even denser formant-rich temporal coverage over the spectral/formant substrate path.

### Phase 1: Vowel-transition family coverage (done)
- Stimulus: 2–3 vowel-transition sequence (e.g., /i/ 0.3s → /ɛ/ 0.3s → /u/ 0.3s)
- Producer must learn to switch vowel targets mid-utterance
- Substrate must track transition context
- Success metric: can the producer match a 2-transition vowel family?

### Phase 2: Temporal Sequences (in progress)
- The longer chains (`long_arc`, `formant_rich`) are now implemented; extend the vowel-transition families to broader temporal sequences
- Producer must switch targets mid-utterance over more steps
- Substrate must retain temporal context over the longer span
- Success metric: can the producer match a 2–3 transition sequence?

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
