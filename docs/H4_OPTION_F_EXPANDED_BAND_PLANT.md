# H4 Option F — expanded-band factorized plant readout

Status: synthetic-only plant confirmation passed on disjoint points. The
[retained result](evidence/2026-07-11-h4-option-f-expanded-band-plant-result.json)
authorizes drafting a separate learner protocol, but not executing learning,
inverse search, recorded audio, or protected-data access.

## Mechanism revision

Option E's five-gesture plant remains unchanged. Option F changes only the
structural sensory readout:

- analytic resonance search expands from `80–3500 Hz` to `80–5000 Hz`;
- the normalized spectral envelope expands from 24 centers at `150–3500 Hz` to
  30 log-spaced centers at `150–5000 Hz`.

This is not a rescue of the failed Option E result. Its `8193–9216` points are
spent and remain failed under their frozen 3.5 kHz gate. Option F uses untouched
five-dimensional Halton indices `12289–13312` and retains the same plant
coefficients, physical solver, target rules, neighborhood tests, and thresholds.

## Evidence basis

The Option E post-hoc decomposition found that all nine P0 failures had a third
separated resonance between `3525` and `3990 Hz`. Geometry and waveform checks
were valid. A motor-domain restriction would discard usable configurations and
would conflict with successful high-constriction vowel targets.

A 5 kHz structural band is still below the 8 kHz Nyquist frequency at the
frozen 16 kHz sample rate. The candidate therefore tests whether the failure
was caused by a truncated sensory representation rather than the plant.

## Advancement boundary

A full plant pass would authorize only a separate preregistration for label-free
babbling and local forward learning using the expanded envelope. It would not
promote H4 or authorize recorded speech. Any failed gate is retained without
changing the band, points, targets, or thresholds.

## Retained outcome

Option F passed P0 and P2 exactly as frozen. All `1024/1024` disjoint states
had three separated resonances through 5 kHz, finite geometry and transfer
values, and deterministic bounded target waveforms. The five selected vowels
all retained identity and `1.0` neighborhood validity. F1/F2/F3 MAE was
`15/56/100 Hz`, and the minimum expanded-sensory separation was `0.4353`
against the `0.25` gate.

This establishes only that the factorized plant and expanded structural readout
provide a viable synthetic motor surface. No forward map was learned and no
perception-to-production inversion occurred.
