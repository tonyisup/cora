# H4 Option H — smooth exemplar-local closure

Status: exact synthetic-only candidate preregistered before implementation or
metrics. Recorded audio and protected data remain unauthorized.

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

