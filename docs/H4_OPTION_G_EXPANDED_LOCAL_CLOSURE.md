# H4 Option G — expanded-band local perception-production closure

Status: exact synthetic-only candidate executed once and retained as failed.
The [result](evidence/2026-07-11-h4-option-g-expanded-local-closure-result.json)
does not authorize a retry, recorded audio, or protected-data access.

## Question

Can label-free sensory-motor babbling learn a bounded local forward map over the
passed five-gesture plant, and can a strictly bounded local inverse recover
unseen synthetic vowel targets well enough to close the perception-production
cycle?

Option F established the plant and 30-value expanded sensory readout. Option G
adds the representation-learning mechanism required by H4:

```text
motor m --physical plant--> expanded sensory S
   |                              |
   +---- label-free babbling -----+
                  |
                  v
        winner-local affine map
                  |
observed S --> bounded local inverse --> motor m' --> physical plant --> S'
```

## Boundaries

- Five vowel labels enter only deterministic target selection and final scoring.
- No label enters babbling, prototypes, fitting, initialization, loss, or search.
- The forward map uses 128 fixed prototypes and winner-local sufficient
  statistics; there is no backpropagation or sequence model.
- The inverse uses at most 21 learned-forward predictions, below the rigid 23
  prediction budget.
- Training, held-out forward evaluation, and inverse targets use disjoint
  five-dimensional Halton ranges not used by Options E or F.
- A pass is synthetic-only evidence for this simplified plant. It is not H4
  promotion, anatomical fidelity, or permission to use recorded speech.

The exact constants and stop rule are authoritative in the companion
[preregistration](evidence/2026-07-11-h4-option-g-expanded-local-closure-preregistration.json).

## Retained outcome

The local map beat neutral, nearest-prototype, and persistence baselines on
`0.965/0.885/0.977` of 512 held-out points, but median unit-vector distance was
`0.264`, above the frozen `0.18` gate. Error was highest in several
high-constriction orthants, and the hard winner-local partition did not meet the
required surface accuracy.

Inverse trials preserved all five vowel identities (`20/20`) and respected all
bounds with at most 20 predictions. That categorical success did not establish
shared geometric closure: combined acoustic/equivalence agreement was `0.40`,
and target-to-final F1/F2/F3 MAE was `16/54/142 Hz`. The result therefore fails
both forward and closure gates. Same-point capacity, threshold, or search-step
tuning is prohibited.

The retained
[post-hoc analysis](evidence/2026-07-11-h4-option-g-failure-analysis.json)
does not select new thresholds. Held-out error correlated with constriction
magnitude (`r=0.415`) and was higher near hard prototype boundaries: mean error
fell from `0.332` in the lowest boundary-margin quartile to `0.265` in the
highest. This supports a future smooth partition-of-unity local mechanism on
disjoint points, not a larger Option G codebook or gate relaxation.
