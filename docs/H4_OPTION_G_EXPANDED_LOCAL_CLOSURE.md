# H4 Option G — expanded-band local perception-production closure

Status: exact synthetic-only candidate preregistered before implementation or
metrics. Recorded audio and protected data remain unauthorized.

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

