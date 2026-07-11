# H4 Option F — expanded-band factorized plant readout

Status: preregistered for one synthetic-only plant confirmation on disjoint
points. No learning, inverse search, recorded audio, or protected-data access is
authorized.

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

