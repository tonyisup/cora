# H4 Option D — geometrically constrained articulatory synthesis

Status: frozen synthetic candidate executed once and retained as a failed
mechanism. The exact protocol is in the
[final preregistration](evidence/2026-07-11-h4-option-d-synthetic-preregistration.json)
and the immutable measurements are in the
[result](evidence/2026-07-11-h4-option-d-synthetic-result.json). No
recorded-human experiment is authorized.

This memo proposes a genuinely different H4 mechanism after the fixed-formant
and continuous acoustic-coordinate candidates failed. It does not change
[VISION.md](../VISION.md) or authorize protected data. The companion
[synthetic preregistration draft](evidence/2026-07-11-h4-option-d-synthetic-preregistration-draft.json)
records the earlier open decisions; the final preregistration closes them for
one synthetic candidate.

## Retained outcome

Implementation commit `62c9535` passed the full test suite before its one
frozen evaluation. Gate 1 passed strongly: held-out median unit-vector sensory
distance was `0.0871` (required at most `0.20`), and the predictor's strict win
fractions against neutral, nearest-prototype, and persistence baselines were
`0.996`, `0.953`, and `0.996` (each required at least `0.75`).

Gate 0 and Gate 2 failed. The plant could not provide a valid selected `/u/`
calibration: its closest frozen grid point had formants `430/1205/2480 Hz` and
was nearest to `/e/` under the preregistered identity rule. Across the 20
vowel-by-F0 closure trials, identity was `0.80`, combined acoustic/equivalence
agreement was `0.60`, and F1/F2/F3 MAE was `7/12/56 Hz`; the required values
were `0.95`, `0.90`, and at most `30 Hz` for each formant. Every `/u/` trial was
reperceived as `/e/`. All physical bounds, loss-improvement checks, label
isolation, and the 23-prediction budget passed; the observed maximum was 17.

Under the frozen stop rule, this is not tunable evidence. The useful retained
result is narrower: the bounded local learner represented the synthetic plant
well on disjoint Halton points, but that did not yield a plant-valid five-vowel
inventory or reliable inverse closure. Recorded speech remains prohibited.

## Decision summary

Option D should represent a motor plan as three bounded **functional gestural
coordinates** controlling a multi-section vocal-tract area function. These
coordinates are inspired by jaw opening, tongue-body constriction, and lip
rounding, but they are not measurements of anatomical muscles or flesh points.

The mechanism separates:

- a deterministic physical plant, which converts geometry into sound; and
- a bounded local forward predictor, which learns motor-to-sensory consequences
  from label-free babbling.

Perception searches the learned forward space for a feasible motor plan.
Production sends that plan through the physical plant. The inverse remains a
search problem, so acoustically equivalent motor solutions are not forced into
one global inverse function.

## Why this is a new H4 mechanism

The fixed-formant route treated five adult F1/F2/F3 table rows as both perception
and production. Option B allowed continuous F0/F1/F2/F3 values, but those were
still unconstrained acoustic coordinates. Its optimizer lowered reconstruction
loss on every recorded development example without producing reliable vowel
identity.

Option D changes the causal direction:

```text
motor gesture m
      |
      v
area function A(x; m) --> acoustic plant g --> waveform --> sensory envelope S
      |                                                   |
      +---------------- babbling observation -------------+
                              |
                              v
                   bounded local predictor f_hat
```

The shared representation is the motor/geometry state used by both the learned
prediction path and the production plant. Formants become consequences to
measure, not motor controls.

## Scope of the first candidate

The first candidate remains vowel-only. It does not attempt consonants, nasal
coupling, respiratory control, vocal-fold biomechanics, syllables, or an infant
vocal tract.

The motor vector is:

\[
m = [m_1,m_2,m_3]^T, \qquad m_i \in [0,1].
\]

| Coordinate | Functional meaning | Honesty boundary |
|---|---|---|
| `m1` | Oral aperture and primary constriction magnitude | Jaw-inspired, not measured jaw displacement |
| `m2` | Longitudinal position of the primary constriction | Tongue-body-inspired, not a tongue flesh-point coordinate |
| `m3` | Lip-end area and effective tract-length control | Rounding-inspired, not measured lip protrusion |

The neutral/rest state `m_rest` must be fixed before execution. The final area
mapping must guarantee positive, finite tube areas and a bounded total tract
length for every point in `[0,1]^3`.

## Physical plant

### Required interface

The authoritative plant is:

\[
g(m,F_0,q) \rightarrow (A,L,H,y),
\]

where:

- `A` contains the cross-sectional areas of the tract sections;
- `L` contains their lengths;
- `H` is the tract transfer function;
- `y` is the synthesized waveform;
- `F0` is an observed or synthetic source fundamental; and
- `q` contains fixed source settings that are not optimized as articulators.

The plant must use more spatial sections than motor dimensions. The recommended
starting point is 8–12 equal-order tubelets controlled by the three gestures.
A literal three-tube plant remains an acceptable educational baseline, but the
project must not describe its three parameters as direct anatomy.

### Area-function mapping

The final preregistration must freeze, numerically:

- the neutral area vector;
- number and length of tube sections;
- how `m1` changes aperture and constriction magnitude;
- how `m2` moves a smooth constriction along the tract;
- how `m3` changes the final sections and effective length;
- minimum and maximum section area;
- wall loss, junction reflection, lip radiation, and termination impedance;
- glottal source shape, F0 handling, attack/release, and output normalization.

A suitable functional form is a neutral log-area function plus a smooth moving
constriction and a lip-end modifier:

\[
\log A_i(m)=\log A_{rest,i}
             + G_i(m_1,m_2)
             + R_i(m_3).
\]

This is a design shape, not a frozen equation. No implementation may begin
until every coefficient and bound is specified in the preregistration.

## Learned local forward predictor

The physical plant and learned model are different objects:

\[
g(m,F_0,q) \rightarrow y, \qquad
\hat f(m,F_0) \rightarrow \hat S.
\]

`g` is known code. `f_hat` is the mechanism learned through babbling. Without
this separation, calling the synthesizer directly would demonstrate search but
not forward-model learning.

### Babbling

The draft candidate uses a deterministic, label-free, low-discrepancy sequence
over `[0,1]^3`. Babbling records only:

```text
(motor command, F0 condition, resulting sensory envelope)
```

Vowel labels and canonical formants do not enter fitting. Training and held-out
motor points must be disjoint. Babbling must also respect a frozen velocity
bound so later trajectory claims are not trained only on discontinuous jumps.

### Bounded local model

The recommended learner is a maximum of 64 locally weighted affine prototypes.
Each prototype may retain:

- a 3-scalar motor center;
- a 24-scalar sensory intercept;
- a `24 × 3` local Jacobian;
- one support/count scalar.

That is at most 100 scalars per prototype and 6,400 learned scalars total.
Only the nearest active prototype updates for one babbling observation. The
prototype replacement rule, activation radius, learning rates, initialization,
and tie order remain unresolved and must be frozen before execution.

The forward-learning gate must evaluate `f_hat` on motor points never used by
babbling. A plant query is ground truth for this synthetic gate; it is not an
extra training observation.

## Sensory representation

The sensory target should represent the vocal-tract envelope rather than raw
harmonic amplitudes. The proposed representation is a 24-value smooth
log-spectral envelope over a frozen voiced-speech band.

Pitch-synchronous harmonics may contribute confidence, but they must not be the
only sampled frequencies. High F0 leaves few harmonics beneath the lower
formants and can make an envelope appear to move with pitch. The recorded
VocalSet domain makes this failure especially relevant.

The sensory loss is:

\[
D_{sensory}
=D_{envelope}
+\lambda_{peak}D_{resonance}
+\lambda_{voice}D_{harmonic}.
\]

- `D_envelope` compares gain-normalized smooth log envelopes.
- `D_resonance` compares localized transfer-function or envelope-derivative
  peaks as a secondary structural term.
- `D_harmonic` weights observations by voiced-source confidence without turning
  pitch harmonics into formant targets.

The window should be 120–200 ms with a 25 ms motor-state hop. A 25 ms analysis
window alone has about 40 Hz FFT spacing at 16 kHz and cannot credibly support a
universal 30 Hz waveform-derived formant gate.

For synthetic data, formants should be measured from the plant transfer
function. Waveform-estimated formants are reported separately.

## Inverse search and redundancy

No global `f_hat^-1` is learned. For each observation, search minimizes:

\[
J(m_t)=D_{sensory}(S_{obs},\hat f(m_t,F_0))
      +\alpha\|m_t-m_{rest}\|_2^2
      +\beta\|m_t-m_{t-1}\|_2^2.
\]

All three components must be normalized before `alpha` and `beta` are selected.
Their values may be selected only inside the synthetic design phase and must be
frozen before the held-out synthetic gate.

The draft search budget is:

1. Initialize from the prior motor state; on the first frame, use the nearest
   stored sensory prototype with rest-effort as the exact tie breaker.
2. Evaluate the initial state once.
3. Search negative then positive offsets for `m1`, `m2`, and `m3`.
4. Run three rounds, halving each step after a complete round.
5. Accept only strict loss improvement; reject out-of-bounds geometry.
6. Use at most `1 + 3 × 3 × 2 = 19` forward predictions per frame, below the
   rigid maximum of 23.
7. Query the physical plant once for final production and reperception.

Starting from `m_(t-1)` selects a stable local solution while allowing other
acoustically equivalent minima to remain represented elsewhere in the forward
map.

## Cycle verification

Exact `||m-m'||` recovery is not the primary criterion because it would punish
valid redundant articulations. The cycle is:

```text
held-out m --> plant g --> observed S --> inverse search --> m'
     ^                                                |
     +----------- compare plant consequences <--------+
```

The synthetic gate must report:

1. **Acoustic cycle agreement** — `g(m')` matches `g(m)` under the frozen
   sensory distance.
2. **Equivalence-aware motor agreement** — motor error is projected onto the
   locally acoustically identifiable subspace of the plant Jacobian, or measured
   against a preregistered acceptable-equivalence set.
3. **Physical validity** — all areas, lengths, velocities, effort terms, and
   synthesized samples remain within bounds.
4. **Post-hoc vowel identity** — the five-vowel labels enter only after each
   final motor plan and waveform are frozen.

The agreement gate is the fraction of held-out trials satisfying the frozen
acoustic and equivalence-aware motor criteria. It must be at least `0.90`.
An unspecified cross-correlation or motor classification score is not enough.

## Synthetic-only evidence ladder

### Gate 0 — plant validity

- every sampled point produces finite positive geometry and stable audio;
- no clipping or unstable filter poles;
- deterministic repeat synthesis is byte- or tolerance-identical;
- control sweeps report area, length, transfer peaks, and output energy;
- plant state and synthesis cost remain bounded.

### Gate 1 — learned forward-map generalization

- babbling and held-out motor points are disjoint;
- every F0 condition appears in both splits without sharing motor points;
- the bounded local predictor beats neutral-tract, nearest-prototype, and
  persistence baselines under the frozen envelope distance;
- report per-region error so pooled performance cannot hide tract-space holes;
- all 64-prototype memory and winner-local update limits hold.

### Gate 2 — held-out synthetic inversion

- at most 23 forward predictions per 25 ms motor hop;
- acoustic/equivalence-aware cycle agreement at least `0.90`;
- clean five-vowel identity preservation at least `0.95`;
- analytic F1, F2, and F3 MAE each at most `30 Hz` for the calibration vowel
  subset proposed by the user;
- also report median and p95 relative formant error, because one absolute bound
  has different meaning for F1 and F3;
- no label enters babbling, forward learning, initialization, or inversion;
- every physical, state, and cost bound passes.

The 30 Hz gate applies only to analytic resonances of the synthetic plant. It
does not apply to 25 ms FFT bins or recorded-human formant estimates.

### Gate 3 — recorded speech remains prohibited

This draft and its first frozen successor are synthetic-only. A full synthetic
pass would permit a separate spent-development preregistration; it would not
authorize VocalSet or CMU execution automatically.

If that later decision is made, only the already-spent development subsets may
be used. VocalSet `female6`, `female7`, `female9`, `male9`, and `male11`; CMU
confirmation speakers `rms`, `awb`, `axb`, `eey`, and `ljm`; and every CMU
reserve speaker remain untouched.

## Stop rules

- Do not implement while the companion draft contains unresolved fields.
- Do not tune area geometry, sensory features, model size, learning rates,
  effort weights, search steps, or gates after held-out synthetic results.
- A failed plant gate permits correction of a code defect, not parameter tuning
  against failed outcomes.
- A failed forward or inverse gate is retained. A retry requires a new mechanism
  preregistration and disjoint synthetic points.
- No synthetic result promotes H4 or establishes infant, human, or DIVA fidelity.

## Research basis

- Story's parametric area-function work supports a neutral tract plus bounded
  spatial modes and length modifications:
  <https://doi.org/10.1121/1.1869752>.
- Arai's sliding three-tube model supports a simple vowel-producing baseline but
  is framed as an educational simplification:
  <https://doi.org/10.1250/ast.27.384>.
- Acoustic-to-articulatory inversion is non-unique; trajectory smoothness is a
  standard way to constrain it:
  <https://pubmed.ncbi.nlm.nih.gov/20968386/>.
- A recent self-supervised imitation agent showed that intelligible output can
  coexist with inconsistent articulatory trajectories, motivating separate
  acoustic and motor-validity gates:
  <https://aclanthology.org/2024.cl-4.5/>.
- High-pitched singing can bias formant estimation through sparse harmonic
  sampling, motivating envelope-first rather than harmonic-only loss:
  <https://doi.org/10.1016/j.specom.2023.103006>.

## Decisions required before freezing

1. Select the plant family: literal sliding three-tube baseline or recommended
   multi-section area function controlled by three gestures.
2. Freeze every area-function coefficient and acoustic solver constant.
3. Freeze the 24-value envelope extractor and all loss weights.
4. Freeze babbling sequence, sample counts, velocity bounds, and held-out split.
5. Freeze prototype initialization, learning rates, activation radius, and
   replacement/tie rules.
6. Freeze rest state, `alpha`, `beta`, coordinate steps, and first-frame
   initialization.
7. Define the equivalence-aware motor threshold and Jacobian rank tolerance.
8. Freeze exact synthetic target motor points and post-hoc vowel mapping.

All eight decisions are resolved for the first synthetic candidate in the final
preregistration. Those values are frozen for implementation and one synthetic
evaluation; they do not authorize recorded speech.
