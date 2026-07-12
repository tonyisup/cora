# Cora-Infans — bounded operational specification

Status: governing source of truth.

This document defines the supported local auditory system, its engineering
limits, and the rules for changing those limits. It supersedes earlier
open-ended or universal research aspirations. It does not supersede retained
preregistrations, results, failure artifacts, or the
[evidence audit](docs/VISION_EVIDENCE_AUDIT_2026-07-11.md); those remain
authoritative for what was actually measured.

## 1. Core mandate

Cora-Infans maintains a deterministic, low-latency local audio substrate that
provides:

- bounded environmental vocal-activity gating under named operating
  conditions;
- bounded bottom-up/top-down predictive tracking;
- exact-efference self-audio verification on physical hardware; and
- a deterministic synthetic vocal-tract plant for infrastructure testing.

Passing a bounded gate must never be generalized beyond its named hardware,
corpus, room, geometry, duration, calibration, and signal conditions.

## 2. Operational stages

### Stage 0 — Substrate architecture

Requirement:

- The software core must use deterministic, bounded pipelines with explicit
  synthetic, offline-real, and live contracts.
- State, iteration counts, playback duration, and memory growth must have
  explicit limits.

Execution boundary:

- Real I/O must fail loudly on missing devices, malformed captures, incomplete
  playback, invalid timestamps, or telemetry discontinuities.
- External-data runners must use retained manifests and hashes. Synthetic
  fallback must never masquerade as real evidence.

### Stage 1 — Real-audio evaluation

Requirement:

- Verification must operate end to end on real frames with auditable manifests,
  source roles, metrics, and development/held-out provenance.
- Third-party material must be licensed for this personal non-commercial use
  and remain outside Git unless redistribution is explicitly permitted.

Constraint:

- The supported evidence is bounded to the retained corpora and captures.
  Broad acoustic spaces, unmodeled rooms, unmapped demographics, and universal
  population claims remain outside validation.

### Stage 2 — Environmental vocal gating

Requirement:

- The vocal front gate may be used as a five-second environmental-control
  filter under the conditions in which it was evaluated.
- Relational cohort geometry may be used only as a batch-calibrated filter when
  the same unlabeled speaker/mode/cohort assumptions are satisfied and reported.

Constraint:

- Reliable generalized cross-cultural IDS/ADS register or parentese detection
  is falsified for the tested absolute, adaptive, temporal, predictive,
  cohort-adaptive, and bounded nonlinear mechanisms.
- Cohort alignment's low false-positive rate is narrow filtering evidence. Its
  poor recall and fieldsite failures prohibit calling it reliable detection,
  cold-start operation, or a causal universal model.
- Active Stage 2 register experimentation is stopped. A deep/global model,
  narrowed claim, same-family retry, or reserved-fieldsite access requires an
  explicit revision to this specification.

### Stage 3 — Local predictive hierarchy

Requirement:

- Maintain bounded local tracking with explicit bottom-up observations,
  higher-level local transition state, and consumed top-down predictions.
- Controlled tone/vowel tracking must retain comparisons against persistence,
  development-unigram, and causal moving-state baselines.

Constraint:

- Predictive support is restricted to the tested controlled streams,
  known-efference dropout, and causal event-state transfer from Beijing to
  Wellington and Mentawai (`MEN`).
- The system does not implement pretrained dialect models. Arbitrary-world
  missingness, unknown dropout timing, semantics, unconstrained acoustic-room
  transfer, and universal language prediction are unsupported.

### Stage 4 — Exact-efference workspace binding

Requirement:

- Bind active audio through a capacity-limited admission and suppression gate.
- Every claimed self identity must have an explicit matching efference identity
  and retained sensory consequence.

Constraint:

- Natural-speech support is locked to an exact commanded-utterance boundary: a
  prior playback replay of that exact utterance is required under the tested
  room, geometry, device, and level conditions.
- Operational reports must identify the command, prior replay, and sensory
  consequence with retained SHA-256 hashes and must carry the bounded workspace
  capacity and clipping count.
- General no-replay self-voice verification, same-source forward
  generalization, arbitrary self spectra, semantics, and geometry transfer are
  unauthorized and unsupported.

### Stage 5 — Synthetic plant infrastructure

Requirement:

- Maintain the factorized five-gesture, ten-section, 5 kHz expanded-band
  transmission-line plant as deterministic software infrastructure.
- Plant geometry, transfer values, target waveforms, and synthetic evidence
  splits must remain reproducible and bounded.

Constraint:

- Continuous motor/perception inverse closure is falsified for the tested
  bounded-local representation families. Active H4 work is stopped.
- Lower spectral reconstruction error or perfect post-hoc vowel identity must
  not be interpreted as continuous perception-production closure, generalized
  speaker identity, anatomical fidelity, or infant motor learning.
- The plant is infrastructure only. A global nonlinear inverse, supervised
  phone model, weakened categorical claim, or local-family retry requires an
  explicit revision to this specification.
- A separate synthetic demonstration may perform bounded selection and recall
  over a finite, previously babbled motor-prototype repertoire under
  [the demo specification](docs/DEMO_SPEC.md). This is authorized only as
  target-specific prototype association and model-based local search. It must
  not be reported as continuous inverse closure or new H4 evidence.

### Stage 6 — Hard efficiency operating point

Requirement:

- On eligible runs using the named callback path, salient
  ADC-frame-end-to-workspace p95 latency must be at most `10.5 ms`.
- Eligible runs must have zero device-deadline misses, overflow flags, callback
  status events, timestamp discontinuities, and fallback frames.
- Active playback intervals must contain zero clipped samples. Whole-capture
  clipping must still be reported separately; clean active playback does not
  erase a trailing-quiet failure.
- Every eligible run must be self-describing: the report must include the exact
  host, input, self-source, world-source, geometry, level, callback backend,
  frame geometry, stimulus hash, and operator confirmations. A passing metric
  without matching operating-point metadata is not an operational pass.

Metric boundary:

- The `10.5 ms` ceiling measures ADC frame end to workspace decision using
  PortAudio callback timestamps. It excludes the approximately `25 ms` frame-
  accumulation interval and does not measure XMOS-internal latency.

Named operating point:

- Host: Mac mini.
- Input: XVF3800 device selector `:0`, `16 kHz` callback capture, two fused host
  channels.
- The XVF3800 has four physical microphones, but this macOS path does not expose
  four raw channels. Host-side four-channel learning or beamforming must not be
  claimed.
- Self source: Mac mini Speakers centered beneath the XVF3800, `4 cm` center to
  center. Exact-efference protocols used `100%` system volume with their own
  frozen digital playback gain.
- External-world source: output device `2`, `180°`, `50 cm`, mechanical knob no
  higher than approximately `50%`. The bounded acceptance profile requires an
  operator confirmation that the knob is at approximately `50%`, not merely
  somewhere below the ceiling; digital playback gain remains protocol-
  specific.

Energy boundary:

- The retained workload measured `27.36 µJ/frame` versus `11.08 µJ/frame` for
  the simpler baseline, a `2.47×` same-host ratio. This is a measured comparison,
  not a permanent energy allowance or wall-plug claim.
- Embodied energy, wall-plug energy, acoustic output efficiency, cross-device
  energy, raw CoreAudio timing, and XMOS-internal latency remain outside
  performance verification.

## 3. Strict governance and provenance

### Protected-data integrity

The following material must remain untouched by automated execution unless a
future specification revision and separate preregistration explicitly authorize
it:

- Music Lab San Diego (`USD`) and Toronto (`TOR`) Stage 2 fieldsites;
- VocalSet `female6`, `female7`, `female9`, `male9`, and `male11`;
- CMU Arctic confirmation speakers `rms`, `awb`, `axb`, `eey`, and `ljm`; and
- CMU Arctic reserve speakers `aew`, `ahw`, `aup`, `fem`, `gka`, `lnh`, `rxr`,
  and `slp`.

Wellington (`WEL`) is protected from new Stage 2 use but is not corpus-pristine:
it was previously consumed by the separate H2 program.

### Frozen-candidate stop rules

- Candidate configuration, data roles, metrics, thresholds, and stop rules must
  be frozen before held-out or confirmation access.
- Missing any mandatory gate freezes that candidate and its spent evidence.
  Parameters, thresholds, representations, or claims must not be adjusted
  retrospectively to rescue it.
- A documented code defect may be corrected only with explicit provenance. If
  candidate metrics already exist, both original and corrected outcomes must be
  retained.
- A genuinely new mechanism requires a new preregistration and disjoint
  evidence. New-mechanism authorization does not authorize protected data.

### Claim and analysis rules

- Post-hoc analysis must be labeled descriptive and must not select a new
  candidate threshold or overwrite a frozen verdict.
- Categorical identity, low false-positive rate, lower reconstruction loss,
  clean active playback, and passing plumbing are narrower facts; none may be
  substituted for a failed broader gate.
- The retained evidence artifacts and
  [project status](docs/PROJECT_STATUS_2026-07-11.md) are authoritative when
  this specification summarizes them.

## 4. Current disposition

- Stages 0 and 1 are implemented and supported within retained boundaries.
- Stage 2 broad register detection and Stage 5 bounded-local closure are
  falsified within tested scope; active work is stopped.
- Stages 3, 4, and 6 are supported only within the named operational boundaries
  above.
- No autonomous expansion beyond these boundaries is authorized by this file.
