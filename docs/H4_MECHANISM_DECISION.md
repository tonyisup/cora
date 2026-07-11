# H4 mechanism decision

Status: Option B approved; development-only continuous inverse synthesis may
proceed under a separate preregistration. Protected confirmation access remains
unauthorized.

This memo does not authorize a new experiment and does not change
[VISION.md](../VISION.md). It records the evidence-backed fork reached after the
fixed-formant route failed in both sung and ordinary read speech.

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

## Prior-art relevance

The older M9 work demonstrates an error-driven perception → production → score
→ update loop and bounded controllers for acoustic imitation. Its vowel-
transition roadmap is useful architectural prior art, but its tone reward and
controller weights are not evidence for recorded-vowel closure here. The useful
part to carry forward is the measured local adaptation loop, not its historical
promotion claims.

## Protected data boundary

Until a decision is made:

- do not fetch VocalSet `female6`, `female7`, `female9`, `male9`, or `male11`;
- do not fetch CMU Arctic confirmation speakers `rms`, `awb`, `axb`, `eey`, or
  `ljm`;
- do not consume the additional CMU Arctic reserve;
- do not run another same-family threshold, representation, or hyperparameter
  retry.

If option B is approved, its first artifact must be a preregistration over
already-spent development data. Approval is not permission to access protected
confirmation audio.
