# H4 Option E — factorized five-gesture tract plant

Status: plant family selected and design search completed. The selected mapping
and one untouched synthetic confirmation are frozen in the
[confirmation preregistration](evidence/2026-07-11-h4-option-e-plant-confirmation-preregistration.json).
No forward learning, inverse search, recorded audio, or protected-data access
is authorized.

## Decision

The next possible H4 plant will retain a bounded one-dimensional transmission
line, but replace Option D's three coupled coordinates with five independently
meaningful functional controls:

\[
m=[j,c,p,l_a,l_p]^T,\qquad m_i\in[0,1].
\]

| Coordinate | Functional control | Why it is separate |
|---|---|---|
| `j` | Oral/jaw aperture | Changes the broader oral cavity without also determining constriction strength |
| `c` | Constriction magnitude | Controls how narrow the primary constriction becomes |
| `p` | Constriction position | Moves that constriction along the tract independently of its magnitude |
| `l_a` | Lip aperture | Controls terminal area independently of tract length |
| `l_p` | Lip protrusion | Controls effective tract length independently of terminal opening |

The names are functional and jaw-, tongue-, and lip-inspired. They are not
measurements of anatomical displacement, muscle activation, or infant vocal-
tract geometry.

## Why this is a new candidate

Option D tied global aperture and constriction magnitude together and tied lip
narrowing to tract length. Its local learner represented that frozen surface,
but the surface itself could not supply a valid `/u/`; the selected `/u/` point
was nearest to `/e/`. It also left F3 MAE at `56 Hz` after inversion.

Option E changes the plant's degrees of freedom before changing any learner:

```text
jaw aperture -------------------+
constriction magnitude ---------+--> area function --> transmission line
constriction position ----------+                         |
lip aperture -------------------+                         v
lip protrusion / tract length --+                       sound
```

This factorization allows a back constriction, small lip opening, and longer
tract to coexist for `/u/`, while allowing front-cavity geometry to vary for
F3. Reusing the Option D coefficients with extra controls would not count as a
new candidate.

## Plant-only evidence ladder

Learning is deliberately excluded from the first protocol.

The frozen [design-only search](evidence/2026-07-11-h4-option-e-plant-design-result.json)
retains all 64 declared coefficient families over Halton indices `4097–5120`.
Family 23 won the preregistered lexicographic selection rule with five valid
vowels, `1.0` minimum neighborhood validity, and worst target relative formant
RMSE `0.1073`. Its numerical mapping is now immutable for confirmation; the
untouched range is `8193–9216`.

### P0 — numerical and physical contract

Every authorized design and confirmation point must have:

- finite positive section areas inside frozen bounds;
- finite tract length inside frozen bounds;
- a finite transfer function and deterministic waveform;
- no synthesized sample at or above full scale; and
- three separated analytic resonances in the scoring band.

### P1 — design-set expressivity

A declared synthetic design set may be used to choose the numerical geometry
coefficients. It is development material, not evidence. Every attempted
coefficient family and its summary must be retained so unsuccessful designs do
not silently disappear.

The design objective is five distinct canonical vowel regions under the
existing post-hoc nearest-vowel rule. Labels may enter this plant-design stage,
because no perceptual learner is being trained. That use must remain explicit.

### P2 — frozen expressivity confirmation

After coefficients are frozen, one untouched deterministic confirmation set
will test:

- valid and distinct `a/e/i/o/u` target regions;
- robustness across all four existing F0 conditions;
- formant error reported separately for F1, F2, and F3;
- local neighborhoods around each selected motor target, so a single isolated
  grid point cannot pass the plant; and
- minimum pairwise separation between the five post-hoc vowel regions.

Failure stops the workflow before babbling, forward learning, or inverse
search. Passing would authorize only a separate learner preregistration—not
recorded speech.

## Separation from the failed evidence

Option D used three-dimensional Halton bases `2,3,5`, training indices
`1–1024`, and held-out indices `2049–2304`. Option E will use five-dimensional
Halton bases `2,3,5,7,11` and indices above `4096`. No Option D target, training,
or held-out motor point will be treated as Option E confirmation evidence.

The exact design/confirmation index ranges, target-selection rule, neighborhood
offsets, formant bounds, and pairwise-separation threshold remain unresolved in
the companion draft. They must be frozen before implementation can generate a
candidate metric.

## Constants intended to remain controlled

Unless the final plant preregistration explains a necessary change, Option E
will retain Option D's:

- ten-section lossy transmission-line solver;
- air density and speed of sound;
- propagation-loss and lip-radiation equations;
- 16 kHz, 160 ms deterministic source and four F0 conditions;
- analytic 5 Hz formant measurement; and
- waveform peak bound.

Keeping these controlled makes the causal revision clear: the factorized area
and length mapping is the new plant hypothesis.

## Stop and honesty rules

- Do not use the Option D result points to choose an Option E pass threshold.
- Do not call design-set success confirmation.
- Freeze all plant coefficients before opening the confirmation set.
- Retain a failed confirmation without coefficient tuning on those points.
- Do not train a forward model unless every plant gate passes.
- Do not claim anatomy, infant fidelity, H4 promotion, or recorded-speech
  readiness from a synthetic plant pass.
- Recorded VocalSet and CMU Arctic data, including every protected speaker,
  remain unauthorized and untouched.

## Decisions required before implementation

1. Freeze the neutral area vector, section count, and area bounds.
2. Freeze separate spatial weights and numerical ranges for `j`, `c`, and `p`.
3. Freeze separate terminal-area and length equations for `l_a` and `l_p`.
4. Freeze the design-set search space and retain-all-attempts report format.
5. Freeze untouched confirmation indices and local-neighborhood construction.
6. Freeze absolute/relative formant, identity, and region-separation gates.
7. Freeze the rule for advancing—or stopping—after the plant result.
