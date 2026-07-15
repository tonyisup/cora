# VISION evidence audit — 2026-07-11

This audit compares the current repository with [VISION.md](../VISION.md). It
does not equate implementation with empirical support and does not reinterpret a
failed gate as partial promotion.

## Stage-by-stage disposition

| Stage | VISION requirement | Authoritative evidence | Audit verdict |
|---|---|---|---|
| 0 — Substrate | Deterministic bounded pipeline, explicit live/offline modes, fail-loud I/O | Full test suite; CLI and adapter contracts | Implemented |
| 1 — Real audio | End-to-end real-frame manifests and auditable held-out metrics | Prior-art replay, Music Lab manifests, recorded WAV/NumPy paths | Passed in bounded licensed corpora; broader conditions remain outside claim |
| 2 — Environmental vocal gating | Five-second vocal front gate plus explicitly batch-bounded cohort filtering | [Final Stage 2 disposition](evidence/2026-07-11-stage2-broad-register-disposition.json) plus retained register results | Vocal front gate remains narrow evidence; broad cross-cultural IDS/ADS detection falsified for tested families |
| 3 — Predictive hierarchy | Bounded local updates, bottom-up/top-down use, held-out advantage, ablations | Natural Wellington/MEN transfer and controlled known-efference dropout results | H2 supported only within tested event-target and known-efference boundaries |
| 4 — Workspace binding | Capacity-limited admission/suppression with flat and fixed-window comparisons | Structured-tone attempt 3 and exact-efference natural-speech spectral-v3 result | H3 supported within exact commanded-utterance boundary; no-replay same-source extension failed |
| 5 — Synthetic plant infrastructure | Deterministic bounded five-gesture, ten-section, expanded-band plant | [Final H4 disposition](evidence/2026-07-11-h4-bounded-local-family-disposition.json) and Options F–H | Plant retained as infrastructure; tested bounded-local inverse family falsified |
| 6 — Efficiency | Named-host latency, RTF, CPU, memory, and measurable energy | Device-timestamp, salient/natural speech, diverse panel, and energy artifacts | H5 supported within measured host/software boundary; embodied and XMOS-internal measurements remain outside claim |

## Requirement audit

### Evidence and governance

- Development, held-out, confirmation, and reserve roles remain explicit in
  retained manifests and preregistrations.
- Failed frozen candidates remain checked in with their original thresholds and
  stop decisions.
- Post-hoc analyses are labeled descriptive and do not replace candidate
  results.
- Broad Stage 2 and bounded-local H4 now have explicit user-selected governance
  artifacts rather than unresolved prose.
- No global/deep replacement, supervised phone model, weakened categorical H4
  claim, or cohort false-positive-control claim is silently promoted.

### Protected material

- Stage 2 San Diego (`USD`) and Toronto (`TOR`) remain pristine; Wellington is
  protected from new Stage 2 use but was previously spent by H2.
- VocalSet `female6`, `female7`, `female9`, `male9`, and `male11` remain
  untouched.
- CMU Arctic confirmation `rms`, `awb`, `axb`, `eey`, and `ljm`, plus the
  documented reserve speakers, remain untouched.
- Third-party audio remains outside Git.

### Verification

- Original audit suite: `487 passed, 3 skipped`. Latest local Python 3.13
  verification after CI retirement: `485 passed, 9 skipped`.
- The skips are intentional hardware/environment-dependent tests.
- `python3 -m cora_infans --operational-readiness` verifies all required
  retained stage evidence against frozen SHA-256 pins and exits nonzero when
  evidence is missing, malformed, replaced, or fails a bounded gate.
- The retained local verification procedure runs readiness before the
  hardware-safe test suite and does not execute live or protected-data runners.
  Remote CI was intentionally retired after this audit for the current
  personal, local-only project scope.
- The self-describing operating-point validator, read-only preflight, and
  explicitly authorized bounded acceptance command close the report/procedure
  gap. After reconnection, the XVF3800 and both output routes passed preflight.
  Attempt 1 is retained as an ineligible low-level failure with zero salient
  frames; its code-defect correction and unchanged retry boundary are recorded
  in `2026-07-12-operational-acceptance-attempt-1.json`.
- Corrected attempt 2 passed the complete self-describing operating point at
  `10.224 ms` salient p95 with `208` salient frames and all required integrity
  counts at zero. Its result is retained and hash-pinned by readiness.
- The later bounded speech-learning demo is governed separately by
  `docs/DEMO_SPEC.md`. Its v3 finite-prototype association passes synthetic
  mirror and cross-configuration panels without changing the stopped H4
  continuous-closure disposition.
- Evidence-link existence and final Stage 2/H4 disposition fields are asserted
  by `tests/test_consolidated_status.py`.
- The working branch is `codex/vision-evidence`.

## Remaining work boundary

No further autonomous empirical run is authorized by the current evidence:

- Stage 2 and active H4 work are stopped by explicit final dispositions.
- Extending H2 to arbitrary-world missingness or semantics requires a new
  observation protocol and broader claim.
- Extending H3 beyond exact efference requires a new generalized-self mechanism
  and likely new physical room/geometry trials.
- Extending H5 to wall-plug/embodied energy or XMOS-internal latency requires
  external instrumentation or firmware-level access.
- Broader Stage 1 demographic/device coverage requires a new licensed-data or
  collection decision.

Those are future user-authorized programs, not unfinished retries of the frozen
work. The repository is internally consistent with its current claim boundary.
