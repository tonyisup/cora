# H4 mechanism decision

Status: final disposition selected. The tested bounded-local H4 representation
family is recorded as falsified within scope, and active H4 experimentation is
stopped. Protected confirmation access remains unauthorized.

The authoritative
[governance artifact](evidence/2026-07-11-h4-bounded-local-family-disposition.json)
records the user's selection of Option 1 after reviewing stop, global-model,
and weakened-claim alternatives.

This memo does not authorize another experiment and does not change
[VISION.md](../VISION.md). It records the evidence-backed fork reached after the
fixed-formant route failed in both sung and ordinary read speech, followed by
the outcome of the approved continuous inverse-synthesis candidate.

## Option B outcome

The candidate was frozen in
[the development preregistration](evidence/2026-07-11-h4-continuous-inverse-synthesis-development-preregistration.json),
implemented at commit `f174243`, and evaluated once on already-spent data. The
[retained result](evidence/2026-07-11-h4-continuous-inverse-synthesis-development-result.json)
failed confirmation readiness:

| Domain | Initial canonical identity | Adapted identity | Required |
|---|---:|---:|---:|
| Clean synthetic | `17/20 = 0.85` | `17/20 = 0.85` | `0.90` |
| VocalSet development | `12/25 = 0.48` | `11/25 = 0.44` | `0.80` |
| CMU Arctic development | `279/500 = 0.558` | `287/500 = 0.574` | `0.80` |

The optimizer behaved as specified: every real-speech example obtained a
strictly lower label-blind spectral reconstruction loss, all plans stayed
within bounds, and no example exceeded 23 syntheses. But lower reconstruction
loss did not transport reliably to the post-hoc vowel identity. Adaptation
changed no synthetic identities, reduced VocalSet identity accuracy, and only
slightly improved CMU Arctic. Motor-plan/reperception agreement was `0.72` and
`0.79` on the two recorded domains, below the required `0.90`; VocalSet's
median relative loss improvement was also only `0.0395`, below `0.10`.

This falsifies this frozen continuous source/filter representation and loss as
a confirmation-ready H4 mechanism. It does not falsify every possible analysis-
by-synthesis model, but another active attempt now requires a genuinely new
mechanism and a new governance decision—not tuning this candidate.
The retained [post-hoc transition analysis](evidence/2026-07-11-h4-continuous-inverse-synthesis-failure-analysis.json)
derives the identity rescue/loss counts from the immutable result without
rerunning or changing the candidate.

## Current contradiction

The present implementation treats five fixed adult F1/F2/F3 points as both the
perceptual category map and the motor coordinate system. It closes perfectly on
its own synthesizer but does not transport to recorded voices:

- the first held-out VocalSet confirmation scored `9/25 = 0.36`;
- five disjoint VocalSet development singers peaked at `0.56` after speaker
  normalization, LPC-order, continuous spectral-to-formant, and centroid
  retries;
- five CMU Arctic development speakers peaked at `286/500 = 0.572`, with `/o/`
  recall `0.13` and `/u/` recall `0.30`.

The failures are consistent across two domains. Fetching either protected
confirmation set would not be credible, and another label-trained phone
classifier would replace analysis by synthesis with supervised recognition.

## Options

### A. Record the fixed-formant route as falsified

Keep H4 open as a long-term hypothesis, but stop active perception-production
work. Preserve clean synthetic closure as a mechanism demonstration only.

This is the smallest and most conservative decision.

### B. Replace the mechanism with continuous inverse synthesis — recommended

Represent a motor plan as bounded continuous source/filter parameters rather
than one of five canonical table rows. Infer those parameters by synthesizing a
candidate waveform, comparing its label-blind spectral envelope and pitch with
the observed waveform, and applying local error-driven parameter updates.

The five vowel labels remain post-hoc evaluation only. The candidate must still
emit the required five-by-five confusion matrix, F1/F2/F3 errors, and clean
synthetic identity score, but its primary mechanism is reconstruction through a
shared generator—not direct phone classification.

The first bounded candidate should freeze:

- continuous F0, F1, F2, and F3 motor coordinates, with fixed safe bandwidths
  and gains unless development evidence preregisters otherwise;
- deterministic local coordinate updates or another explicitly local search;
- bounded iteration count, state, and synthesis cost;
- reconstruction improvement over both the unadapted canonical motor plan and
  a nearest-spectral-centroid classifier;
- a perception → motor-plan → synthesis → perception loop, with identity and
  reconstruction reported separately;
- development-only use of already-spent VocalSet and CMU Arctic material;
- no access to either protected confirmation set until a frozen development
  candidate reaches the existing credibility gates.

This changes the implementation substantially, but it better matches H4's
original shared acoustic-articulatory and motor-adaptation language than the
current fixed lookup table.

### C. Adopt a supervised phone model or deep/global encoder

This could improve recognition accuracy, but it materially changes the project
from local analysis by synthesis to supervised or globally trained phone
classification. It should require a VISION revision before implementation.

### D. Geometrically constrained articulatory synthesis — retained failure

Represent three bounded functional gestures as controls over a physical
multi-section vocal-tract area function. Learn a separate bounded local
motor-to-sensory predictor through label-free babbling, then invert observations
by locally searching the learned forward space. Production uses the physical
plant; perception and production share its motor/geometry state.

This is genuinely different from Option B because F1/F2/F3 are measured
consequences rather than motor coordinates. The proposal, honesty boundaries,
equivalence-aware cycle metric, synthetic gates, and unresolved freeze decisions
are in [the Option D design memo](H4_OPTION_D_GEOMETRIC_ARTICULATORY_SYNTHESIS.md).
Its earlier
[preregistration draft](evidence/2026-07-11-h4-option-d-synthetic-preregistration-draft.json)
preserves the open design state. The user subsequently authorized implementation,
and the exact [synthetic-only preregistration](evidence/2026-07-11-h4-option-d-synthetic-preregistration.json)
closes every field. It does not authorize recorded audio.

The candidate was implemented at commit `62c9535` and evaluated exactly once.
The retained [synthetic result](evidence/2026-07-11-h4-option-d-synthetic-result.json)
passed its local-forward gate: median held-out sensory distance was `0.0871`,
and strict win fractions against neutral, nearest-prototype, and persistence
baselines were `0.996/0.953/0.996`. The full candidate nevertheless failed:

- the plant's selected `/u/` calibration mapped post hoc to `/e/`, so Gate 0
  failed;
- inverse identity was `0.80` rather than `0.95`, with every `/u/` reperceived
  as `/e/`;
- acoustic-plus-equivalence agreement was `0.60` rather than `0.90`;
- F1/F2/F3 MAE was `7/12/56 Hz`, so F3 missed the `30 Hz` bound.

All motor/geometry and search-cost bounds passed, with at most 17 learned
forward predictions per trial. No recorded audio or protected data was
accessed. The result supports the narrow engineering proposition that a
winner-local predictor can approximate this synthetic motor-to-sensory surface;
it does not establish perception-production closure for the frozen plant.

## Prior-art relevance

The older M9 work demonstrates an error-driven perception → production → score
→ update loop and bounded controllers for acoustic imitation. Its vowel-
transition roadmap is useful architectural prior art, but its tone reward and
controller weights are not evidence for recorded-vowel closure here. The useful
part to carry forward is the measured local adaptation loop, not its historical
promotion claims.

## Protected data boundary

Following the failed Option B candidate:

- do not fetch VocalSet `female6`, `female7`, `female9`, `male9`, or `male11`;
- do not fetch CMU Arctic confirmation speakers `rms`, `awb`, `axb`, `eey`, or
  `ljm`;
- do not consume the additional CMU Arctic reserve;
- do not run another same-family threshold, representation, or hyperparameter
  retry.

The completed Option B workflow followed a preregistration over already-spent
development data. Its failure is not permission to access protected
confirmation audio.

## Subsequent plant direction

After retaining the Option D failure, the user selected an expanded
one-dimensional area-function plant as the next design direction. The
[Option E memo](H4_OPTION_E_FACTORIZED_FIVE_GESTURE_PLANT.md) separates jaw
aperture, constriction magnitude, constriction position, lip aperture, and lip
protrusion. Its first protocol is plant-only: expressivity must pass before any
babbling or inverse learner is considered.

The companion
[preregistration draft](evidence/2026-07-11-h4-option-e-five-gesture-plant-preregistration-draft.json)
contains unresolved coefficients and gates and is not execution-ready. This
design decision does not authorize a candidate run or recorded audio.

That draft was subsequently resolved through a retained 64-family design-only
search and an untouched
[plant confirmation](evidence/2026-07-11-h4-option-e-plant-confirmation-result.json).
Option E passed its five-vowel expressivity gate, including `/u/`, but failed
the required all-state plant contract because `9/1024` arbitrary motor states
did not expose three separated resonances in the frozen analysis band. The
result therefore does not authorize the planned local learner. No recorded or
protected data was accessed.

The retained failure decomposition located every missing third peak between
`3525` and `3990 Hz`, rather than finding invalid geometry. Option F therefore
freezes an expanded `80–5000 Hz` structural readout on disjoint points while
leaving the plant unchanged. Its
[preregistration](evidence/2026-07-11-h4-option-f-expanded-band-plant-preregistration.json)
authorizes one synthetic plant-only confirmation, not learning or recorded
speech.

The [Option F result](evidence/2026-07-11-h4-option-f-expanded-band-plant-result.json)
passed both frozen gates: `1024/1024` states exposed three resonances, all five
vowel regions and neighborhoods retained identity, and F1/F2/F3 MAE was
`15/56/100 Hz`. This supports proceeding to a separately frozen label-free
local learner. It does not by itself support H4 or recorded-speech access.

The exact next mechanism is frozen as
[Option G](H4_OPTION_G_EXPANDED_LOCAL_CLOSURE.md): a 128-prototype winner-local
affine map learned from label-free five-dimensional babbling, followed by a
21-prediction bounded inverse. Its disjoint
[preregistration](evidence/2026-07-11-h4-option-g-expanded-local-closure-preregistration.json)
must pass both forward generalization and perception-production closure before
any recorded-speech discussion.

The [Option G result](evidence/2026-07-11-h4-option-g-expanded-local-closure-result.json)
failed both gates. It preserved vowel identity perfectly and beat all three
forward baselines frequently, but median forward distance was `0.264`, combined
agreement `0.40`, and F1/F2/F3 reconstruction MAE `16/54/142 Hz`. This supports
neither H4 promotion nor a same-candidate retry.

The Option G failure analysis supports one distinct local alternative:
[Option H](H4_OPTION_H_SMOOTH_LOCAL_CLOSURE.md) removes hard regions and fits a
Gaussian-weighted affine tangent from the 64 nearest babbling exemplars at each
query. Its [preregistration](evidence/2026-07-11-h4-option-h-smooth-local-closure-preregistration.json)
uses entirely disjoint points and retains the same closure gates.

The [Option H result](evidence/2026-07-11-h4-option-h-smooth-local-closure-result.json)
also failed. Median forward distance worsened to `0.293`, the smooth model beat
nearest-exemplar on only `0.479` of points, combined agreement remained `0.40`,
and F1/F2/F3 MAE was `15/90/163 Hz`. Perfect vowel identity in both Options G
and H is insufficient because the continuous shared representation did not
close.

This reaches a governance boundary. Another local capacity, density, kernel,
or search retry would tune the failed family. A global nonlinear encoder,
supervised articulatory inversion, or relaxed continuous-closure claim would
materially revise [VISION.md](../VISION.md) and requires an explicit user
decision. Active H4 experimentation stops here meanwhile.

## Final disposition

The user selected: **record bounded-local H4 as falsified**.

The project therefore concludes that its tested fixed-formant, continuous
acoustic-coordinate, hard winner-local, and smooth exemplar-local mechanisms do
not satisfy the continuous H4 perception-production closure target. Perfect
post-hoc vowel identity in Options G and H does not override their failed
forward, equivalence, and formant gates.

The successful Option F factorized plant/readout remains reusable synthetic
infrastructure, but it is not evidence that H4 closed. No further local
capacity, density, kernel, threshold, or search retry is authorized. A global
nonlinear mechanism or weaker categorical hypothesis would require a future
explicit VISION revision. Protected VocalSet and CMU material remains untouched.
