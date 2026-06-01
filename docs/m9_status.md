# M9 Sound-First Motor Learning — Status

> Status: Vowel imitation working on hardware. Live mic → vowel classification → formant synthesis.

## What Works

### Vowel Imitation (main path)
- **5 vowels**: /i/, /a/, /u/, /E/, /c/ with distinct formant qualities
- **Live mic input**: Real-time classification + synthesis through headphones
- **Zero warble**: Lookup table approach eliminates frame-to-frame jitter
- **Phase-continuous synthesis**: Smooth across frame boundaries

### Architecture
- **Substrate**: RNN (32-dim), trained on vowel audio (`data/m9_substrate_formant.ckpt.json`)
- **Classifier**: Nearest-centroid lookup (`data/m9_formant_lookup_v3.json`)
- **Synthesizer**: 3-formant oscillator with phase continuity + HP filter on input

### Key Files
| File | Purpose |
|------|---------|
| `scripts/m9_live_vowel.py` | Live mic → vowel imitation (main demo) |
| `scripts/m9_formant_demo.py` | Offline vowel demo with comparison |
| `scripts/m9_formant_stimuli.py` | Vowel stimulus generator |
| `scripts/m9_temporal_demo.py` | Pure tone temporal sequences |
| `data/m9_substrate_formant.ckpt.json` | Vowel-trained substrate |
| `data/m9_formant_lookup_v3.json` | Vowel lookup table (centroids + templates) |

### Active Checkpoints
```
data/m9_substrate_formant.ckpt.json    # Vowel substrate (32-dim RNN)
data/m9_substrate_live.ckpt.json       # Pure tone substrate (temporal path)
data/m9_controller_temporal_v3.ckpt.json  # Temporal controller (pure tones)
data/m9_formant_lookup_v3.json         # Vowel lookup table
data/m9_temporal_demo.py               # Uses temporal_v3 + substrate_live
```

### Archived (superseded) Checkpoints
```
data/archive/m9_controller_*.ckpt.json  # Abandoned NN controller variants
data/archive/m9_substrate_*.ckpt.json   # Abandoned substrate variants
scripts/archive/m9_train_*.py           # Superseded training scripts
```

### Key Bugs Fixed
1. **Playback trigger**: Exact equality → flag pattern
2. **Phase discontinuity**: Independent φ per frame → cumulative phase tracking
3. **Room noise**: Added 100 Hz HP filter on input
4. **Controller instability**: Replace NN with lookup table
5. **Naming mismatch**: Consolidated to clear substrate/live split

### Known Limitations
- Vowel classification jumps between frames (clusters overlap in hidden state space)
- Works best with sustained vowels (>0.5s)
- 5 vowels distinguishable, but /i/ and /E/ sometimes confused
- Mic sensitivity varies — threshold may need tuning per environment

### Next Steps
- Improve substrate training for better vowel cluster separation
- Add more vowels to lookup table
- Text-vowel association (word → sound mapping)
