# H4 Option H — smooth exemplar-local closure

Status: exact synthetic-only candidate executed once and retained as failed.
The [result](evidence/2026-07-11-h4-option-h-smooth-local-closure-result.json)
closes this smooth-local branch; recorded audio and protected data remain
unauthorized.

Option H replaces Option G's discontinuous hard-winner affine partition with a
smooth locally weighted affine predictor. It retains a bounded label-free
babbling memory and fits only the 64 motor-nearest exemplars at each query using
Gaussian distance weights. There is no global backpropagation, sequence model,
or vowel supervision.

This is a mechanism change, not a capacity retry: Option G's 128 hard regions
and accumulated coefficients are removed. Option H stores a fixed 2,048-point
motor/sensory babbling table and constructs a query-local tangent estimate.
Disjoint training, forward, and inverse ranges prevent the spent Option G points
from tuning or rescoring the candidate.

The exact memory, kernel, ridge, initialization, search, gates, and stop rule are
frozen in the companion
[preregistration](evidence/2026-07-11-h4-option-h-smooth-local-closure-preregistration.json).

## Retained outcome

Smooth local regression did not fix Option G. Median held-out sensory distance
was `0.293` against `0.18`, and it beat the nearest stored exemplar on only
`0.479` of points against `0.75`. It still beat neutral and persistence on more
than `0.96` of points, confirming that the babbling memory contains useful
structure but that this local tangent estimator does not reconstruct it
accurately enough.

All `20/20` inverse trials retained vowel identity and stayed within bounds and
the 23-prediction budget. Continuous closure again failed: combined agreement
was `0.40`, and F1/F2/F3 MAE was `15/90/163 Hz`. Kernel, neighbor-count,
babbling-density, and search retries on these spent points are prohibited.
