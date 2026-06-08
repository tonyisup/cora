# M9 Sound-First Motor Learning — Status

_Updated: 2026-06-04_

> Status: 5-vowel spectral imitation on hardware. Text→vowel association working.

## What Works

### Vowel Imitation (main path)
- **5 vowels**: /i/ (beat), /ɛ/ (bet), /ɑ/ (hot), /ɔ/ (caught), /u/ (boot)
- **Validated suite:** 604 passed, 1 skipped, 0 failed
- **Mel spectral substrate**: 26-bin Mel filterbank → RNN (32-dim) → nearest-centroid lookup
- **100% synthetic accuracy**, separation ratio 3.39 (3 vowels) / 1.21 (5 vowels)
- **Live mic input**: Real-time classification + synthesis through headphones
- **Zero warble**: Lookup table approach eliminates frame-to-frame jitter
- **Phase-continuous synthesis**: Smooth across frame boundaries
- **Temporal smoothing**: 3-frame majority vote
- **Speech-gated RNN**: Only feeds frames above threshold, avoids silence contamination
- **Confidence scoring**: 2nd-nearest centroid distance ratio

### Text → Vowel Association
- **Pronunciation dictionary**: 75+ words mapped to vowel sequences
- **Word → vowel → formant synth**: Type a word, hear its vowel
- **Multi-vowel words**: `hello` → /ɛ/-/u/, `cora` → /ɔ/-/ɑ/

### Temporal Smoke Harness
|- **Saved temporal controller/substrate pair**: hardware-free smoke run over a 3-tone sequence
|- **Frame continuity check**: cumulative phase rendering verified by regression harness
|- **Noisy-room regression**: additive input noise can be injected deterministically and still preserve the decode
|- **Diphthong-family coverage**: a 5-tone transition family exercises longer temporal sequences
|- **Wide-sweep family coverage**: a 7-tone transition family expands the transition span further
|- **Device-profile regression**: high-pass/gain variation stays stable under the smoke harness
|- **Family-pair sweep**: ordered pair sweep reports primary/held-out rewards and sweep-derived blend weights
|- **Trainer**: `scripts/m9_temporal_train.py` now scores in-memory weights locally and uses sweep-derived held-out weighting by default
|- **CLI**: `scripts/m9_temporal_sequence_smoke.py` (supports `--output-wav` for a listenable rendered artifact)

### Architecture
- **Substrate**: RNN (32-dim), Mel spectral input, contrastive triplet-trained
- **Classifier**: Nearest-centroid in 32-dim hidden state space
- **Synthesizer**: 3-formant oscillator with phase continuity + HP filter on input
- **Training**: Adam optimizer with gradient clipping, triplet margin loss, desaturation penalty, analytic BPTT

### Key Files
| File | Purpose |
|------|---------|
| `scripts/m9_live_vowel.py` | Live mic → vowel imitation (main demo) |
| `scripts/m9_vowel_say.py` | Text → vowel association (type words, hear vowels) |
| `scripts/m9_train_spectral.py` | Contrastive spectral substrate training |
| `scripts/m9_train_substrate_contrastive.py` | Raw-audio contrastive training (superseded) |
| `scripts/m9_temporal_sequence_smoke.py` | Hardware-free temporal-sequence smoke harness |
| `scripts/m9_temporal_train.py` | In-memory temporal trainer with sweep-derived held-out weighting |
| `scripts/m9_temporal_family_sweep.py` | Ordered family-pair sweep and generalization report |
| `docs/m9_temporal_training_workflow.md` | Reusable note for the promoted temporal-training workflow |

### Active Checkpoints
```
data/m9_substrate_spectral.ckpt.json   # Spectral substrate (Mel input, 1,888 params)
data/m9_formant_lookup_v5.json         # 5-vowel lookup table
data/m9_substrate_live.ckpt.json       # Pure tone substrate (temporal path, legacy)
data/m9_controller_temporal_v3.ckpt.json  # Temporal controller (pure tones, legacy)
```

### Archived (superseded) Checkpoints
```
data/archive/m9_controller_*.ckpt.json  # Abandoned NN controller variants
data/archive/m9_substrate_*.ckpt.json   # Abandoned substrate variants
data/m9_substrate_formant.ckpt.json     # Raw-audio substrate (pre-contrastive)
data/m9_substrate_formant_v2.ckpt.json  # Raw-audio contrastive substrate (superseded by spectral)
data/m9_formant_lookup_v*.json          # Earlier lookup tables
scripts/archive/m9_train_*.py           # Superseded training scripts
```

### Key Bugs Fixed
1. **Playback trigger**: Exact equality → flag pattern
2. **Phase discontinuity**: Independent φ per frame → cumulative phase tracking
3. **Room noise**: Added 100 Hz HP filter on input
4. **Controller instability**: Replace NN with lookup table
5. **Cluster overlap**: Contrastive triplet training pushes vowel embeddings apart
6. **RMS norm mismatch**: Training/deployment normalization must match exactly
7. **Information loss**: Per-frame RMS norm destroyed F2 cues → Mel spectral input preserves them
8. **Silence contamination**: Speech-gated RNN skips silent frames

### Known Limitations
- /i/ and /ɛ/ sometimes confused on live mic in noisy rooms
- Mic sensitivity varies — threshold may need tuning per environment
- /ɪ/ (bit) and /ʊ/ (book) excluded — too close to /i/ and /u/ for 3-formant modeling
- Temporal harness currently uses pure-tone families; vowel-transition family over the spectral substrate is the next step

### Next Steps
- Vowel-transition family coverage over the spectral substrate path
- Longer temporal sequences with real formant transitions
- Expand pronunciation dictionary
- Consonant frame modeling (fricatives, stops, nasals)
- Substrate retraining on live mic audio (reduce synthetic-domain gap)
