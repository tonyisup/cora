# Project Status — 2026-07-09

## Current focus

Cora-Infans is moving from simulation-only self-prediction/corollary-discharge
work into an og-cora M8-inspired speaking-policy ladder. The guiding decision is
explicit: port the *policy and acceptance-gate shape* before adding any live
speaker path.

## Live-playback boundary

Live playback remains disallowed by default. Any future live SPEAK milestone must
be gated behind both:

1. the simulation-only self-forward-model promotion gate, and
2. the simulation-only speaking-policy promotion gate.

The current M8-infans work is offline only: no TTS, no playback, no simultaneous
speaker/mic loop, and no new audio dependencies.

## M8-infans offline speaking-policy ladder begins

**Decision:** port the *policy/acceptance shape* of og-cora M8 before any live
speaker path. In cora-infans this means an offline, deterministic ladder for
conversation state → speaking action → reward, while preserving the explicit
`live_playback=false` boundary.

**What was added:**
- `docs/plans/2026-07-09-live-speaking-policy-from-m8.md`: implementation plan
  grounded in `/Users/juicebox/Projects/cora/.worktrees/og-cora/docs/m8_speaking_policy.md`.
- `src/cora_infans/speaking_policy.py`: `ConversationFeatures`, the explicit
  action set (`speak`, `wait`, `yield`, `repair`, `clarify`), and
  `HeuristicSpeakingPolicy`.
- `src/cora_infans/conversation_tracker.py`: pure observer from scripted
  conversational events to policy-facing features.
- `src/cora_infans/speaking_reward.py`: deterministic M8-style reward terms for
  repair, yield, clarification, smooth turn transitions, and stalls.
- `src/cora_infans/speaking_simulator.py`: offline scripted episode replay that
  composes tracker + policy + reward and emits JSONable reports with
  `live_playback=false` and `live_playback_allowed=false`.

**Honesty boundary:** this is a speaking-policy scaffold, not speech execution.
It does not synthesize, play, or capture live speech. The next live milestone
still requires a separate SPEAK-state/mic-suppression design and must remain
gated behind both self-forward-model promotion and speaking-policy promotion.

**Validation:** focused tests cover every heuristic action branch, tracker state
transitions, reward terms, and deterministic simulator replay. Full suite:
`pytest tests -q` → 163 passed.

## M8-infans-D — Speaking-policy promotion gate

**Decision:** add a single simulation-only promotion verdict for the offline
speaking-policy ladder, analogous to the self-forward-model promotion gate. This
keeps live playback gated behind an auditable action-policy checkpoint rather
than a loose collection of episode reports.

**What was added:**
- `canonical_speaking_policy_episodes()` in `src/cora_infans/speaking_simulator.py`:
  deterministic fixtures for clean handoff, overlap/yield/repair, clarification,
  and quiet no-false-action cases.
- `run_speaking_policy_promotion_gate(...)`: evaluates canonical episode reports
  against five criteria:
  1. reward ratio meets threshold,
  2. active overlap yields,
  3. no false repairs,
  4. no unnecessary clarification,
  5. no excessive stall penalty.
- CLI: `python -m cora_infans --promote-speaking-policy [--min-reward-ratio F]`.
- `--describe` now reports `simulation_only_speaking_policy_promotion_gate`.

**Output shape:**
```json
{
  "mode": "simulation_only_speaking_policy_promotion_gate",
  "verdict": "promote_simulation_only",
  "live_playback": false,
  "live_playback_allowed": false,
  "criteria": {
    "reward_threshold": {"passed": true},
    "overlap_yields": {"passed": true},
    "no_false_repairs": {"passed": true},
    "no_unnecessary_clarification": {"passed": true},
    "no_excessive_stall_penalty": {"passed": true}
  }
}
```

**Honesty boundary:** passing this gate promotes only the offline policy scaffold.
It still does not authorize live TTS or speaker output. Live playback remains
behind both self-forward-model promotion and speaking-policy promotion, plus a
separate SPEAK-state/mic-suppression design.

**Validation:** focused gate and CLI tests pass; full suite: `pytest tests -q` →
166 passed.

## M8-infans-E — Combined speaking readiness gate

**Decision:** join the speaking-policy promotion gate with the self-forward-model
promotion gate before any simulated speak authorization. This makes the M8 bridge
explicit: the agent may only mark `simulated_speak_authorized=true` when both
conversation action selection and self-prediction safety have passed.

**What was added:**
- `run_speaking_readiness_gate(...)` in `src/cora_infans/speaking_simulator.py`:
  combines `run_speaking_policy_promotion_gate(...)` with
  `run_forward_model_promotion_gate(...)`.
- CLI: `python -m cora_infans --promote-speaking-readiness
  [--min-speaking-reward-ratio F] [--min-self-world-correlation F]`.
- `--describe` now reports `simulation_only_speaking_readiness_gate`.

**Output shape:**
```json
{
  "mode": "simulation_only_speaking_readiness_gate",
  "verdict": "promote_simulation_only",
  "simulated_speak_authorized": true,
  "live_playback": false,
  "live_playback_allowed": false,
  "criteria": {
    "speaking_policy_promoted": {"passed": true},
    "self_forward_model_promoted": {"passed": true},
    "live_playback_blocked": {"passed": true}
  }
}
```

**Honesty boundary:** passing this gate still does not authorize live playback.
It authorizes only simulation-side SPEAK decisions. The next milestone is a design
doc for live SPEAK state, mic suppression/tagging, latency measurement, and
feedback safety.

**Validation:** focused readiness and CLI tests pass; full suite: `pytest tests -q`
→ 170 passed.

## M8-infans-F — Live SPEAK-state and mic-suppression design doc

**Decision:** stop at the live-playback boundary and document the required safety
architecture before any speaker/mic implementation. The design keeps existing
live listening safe and requires a separate explicit command for any future live
SPEAK smoke.

**What was added:**
- `docs/plans/2026-07-09-live-speak-state-and-mic-suppression.md`.
- Proposed SPEAK states: `IDLE`, `READY_TO_SPEAK`, `SPEAKING`,
  `POST_SPEAK_SUPPRESSION`, `RECOVERING`.
- Required mic-frame tags: `speak_state`, `speaker_output_active`,
  `efference_id`, `expected_self_audio`, `self_suppression_applied`,
  `latency_aligned_ms`, `contamination_policy`, and `usable_for_world_learning`.
- First live-slice recommendation: a bounded explicit `--speak-live-smoke`
  harness with injectable fake playback/capture tests, defaulting to `tag_only`.

**Honesty boundary:** the doc explicitly forbids overloading `--listen-live` or
adding automatic playback. Passing simulation gates authorizes only simulated
speaking; real hardware output still requires explicit opt-in and bounded smoke
validation.

**Validation:** documentation-only change; no runtime behavior changed.

## M8-infans-G — Fake-I/O bounded SPEAK smoke helper

**Decision:** start implementation after the SPEAK-state design with a pure helper
and injected fake I/O only. This creates the report/artifact/tagging surface
needed for future live validation without touching speakers or microphones.

**What was added:**
- `src/cora_infans/live_speak_smoke.py` with `run_live_speak_smoke_trial(...)`.
- `tests/test_live_speak_smoke.py` covering:
  - efference WAV creation,
  - captured WAV creation through injected fake capture,
  - required SPEAK-state frame tags,
  - safe blocked default when playback/capture functions are not injected.

**Current behavior:**
- With injected fake functions, the helper returns `verdict=fake_io_smoke_complete`.
- Without injected functions, it returns `verdict=blocked_missing_injected_io` after
  writing only the planned efference WAV.
- `real_hardware=false`, `live_playback=false`, and `live_playback_allowed=false`
  remain hard-coded in this slice.

**Honesty boundary:** this is not a live command and does not perform playback.
It only establishes the state/tag/artifact contract for the future explicit live
smoke harness.

**Validation:** focused fake-I/O tests pass; full suite: `pytest tests -q` →
172 passed.

## M8-infans review — logic defects fixed (#1–#4)

**Trigger:** post-implementation review of M8-A..G found four logic gaps. All fixes
are simulation-only; `live_playback`/`live_playback_allowed` remain `False`.

**#1 Collision leak (bug):**
- `HeuristicSpeakingPolicy.decide` only guarded `speak` when `is_agent_speaking and was_collided` or `was_interrupted and has_deferred_output`. An `overlap_detected` emitted without a prior `agent_speech_start` left both false and fell through to `speak` into the collision.
- Reward's `missed_repair` also gated on `has_deferred_output`, so the leak was unpenalized.
- Fix: collision/interrupt now suppresses `speak` directly (`wait`/`yield`/`repair` per state); `missed_repair` applies to any non-repair action while repair is needed.
- Tests added: `test_collision_without_agent_speech_does_not_speak`, `test_speaking_into_an_unrepaired_collision_is_penalized`.

**#2 Reward/policy contradiction (bug):**
- Policy default speaks at silence >= 2000 ms, but `smooth_turn_transition` only fired for silence <= 1500 ms, so the happy-path `speak` never earned its bonus and the canonical `clean_turn_handoff` contributed 0 reward.
- Fix: raised the reward window to silence <= 2000 ms so the policy's own default turn handoff is rewarded.
- Test added: `test_smooth_turn_transition_at_policy_speak_threshold`.

**#3 Magic acceptance constant (robustness):**
- `run_speaking_policy_promotion_gate` compared total canonical reward against a hard-coded `target_reward = 4.0`, unrelated to the episodes.
- Fix: `target_reward` is now the summed canonical episode reward, documented as an explicit derived acceptance threshold.
- Test added: `test_promotion_target_reward_is_derived_from_canonical_episodes`.

**#4 Degenerate frame tag (cosmetic):**
- `live_speak_smoke._frame_tags` could produce a zero-width `RECOVERING` tag when capture barely exceeded the speak window.
- Fix: `RECOVERING` always gets positive width; `POST_SPEAK_SUPPRESSION`/`RECOVERING` split clamped.
- Test added: `test_recovering_frame_tag_has_positive_width_at_boundary`.

**Validation:** full suite: `pytest tests -q` → 177 passed. Promotion gate verdict remains `promote_simulation_only` with `reward_ratio=1.0` (target derived from canonical sum).

## M8-infans-H — Safe-by-default SPEAK smoke CLI

**Decision:** expose the bounded SPEAK smoke helper on the CLI, but keep it safe by
default. No real playback exists; the command name makes risk explicit.

**What was added:**
- `speak_live_smoke_cli(...)` in `src/cora_infans/cli.py`.
- New command:
  ```
  python -m cora_infans --speak-live-smoke --out-dir DIR [--vowel a|e|i|o|u]
    [--duration-ms N] [--max-capture-ms N] [--fake-io]
  ```
- `--describe` now reports `bounded_live_speak_smoke`.

**Safety behavior:**
- Without `--fake-io`: writes only the planned efference WAV, returns
  `verdict=blocked_missing_injected_io`, `captured_wav=None`.
- With `--fake-io`: runs the in-process fake playback/capture harness (silent
  captured frames), returns `verdict=fake_io_smoke_complete`.
- In both cases `real_hardware=false`, `live_playback=false`, and
  `live_playback_allowed=false` are hard-coded. There is no real-speaker path.

**Honesty boundary:** the command is a smoke harness, not live playback. Real
hardware validation remains deferred behind explicit user opt-in and the SPEAK
design doc.

**Validation:** new CLI tests pass (blocked default + fake-io path); full suite:
`pytest tests -q` → 179 passed.

## Next step

M8 is now a coherent simulation-only ladder: policy → tracker → reward → simulator
→ promotion gate → readiness gate → bounded SPEAK smoke (fake-I/O, no playback).
The remaining real boundary is the live SPEAK design execution, which must wait
for explicit user opt-in. Suggested next work is strengthening the canonical
episode coverage (e.g. backchannel, barge-in recovery) or a deterministic
latency cross-correlation stub for the future real-hardware slice.

## M8-infans coverage hardening — canonical episodes + latency stub

**Decision:** strengthen the canonical episode set (backchannel suppression,
barge-in recovery) and pre-build the deterministic latency cross-correlation
estimator needed for the future real-hardware slice, without adding any live path.

**Episode additions (`canonical_speaking_policy_episodes`):**
- `barge_in_recovery`: user interrupts the agent mid-speech. Policy protects the
  user's speech (wait), then repairs the deferred/interrupted output after the user
  ends. Exercises deferred-output repair under barge-in.
- `backchannel_suppression`: user is speaking, agent has queued output. Policy must
  not speak over the user (wait) and only speak after silence. Exercises
  backchannel suppression (no premature turn takeover).

**Reward/policy correction found while wiring episodes:**
- `missed_repair` previously penalized *any* non-repair while a collision/interrupt
  was unresolved — including the correct "wait" while the user is actively speaking.
  Fixed: `missed_repair` no longer fires when `is_user_speaking` (yielding the floor
  is correct, not a missed repair).
- Promotion criterion `overlap_yields` required every collision step to be `yield`,
  which failed for a barge-in where the user is mid-speech (correct action `wait`).
  Renamed to `collision_safety` and now requires only "no speak into collision"
  (every overlap step is non-`speak`).

**Latency cross-correlation stub (`live_speak_smoke.estimate_playback_latency_ms`):**
- Pure deterministic numpy normalized cross-correlation over a bounded lag window.
- Recovers injected sample delay on synthetic PCM (verified in tests within ±2 ms).
- Explicitly NOT wired to any playback/capture path. The `--real-io` flag that
  would call it does not exist; this function never triggers speaker/mic I/O.

**Honesty boundary:** both additions are simulation-only. No live playback path was
opened; `live_playback_allowed` stays `False` everywhere.

**Validation:** full suite: `pytest tests -q` → 183 passed. Promotion gate still
`promote_simulation_only` (6 episodes, `reward_ratio=1.0`, `collision_safety=1.0`).

## M8-infans real-I/O slice — gated `--real-io` SPEAK smoke

**Decision (user-approved 2026-07-09):** implement the real-hardware opt-in path
from the M8-F plan, gated behind the speaking-readiness gate. The user explicitly
confirmed readiness for actual audio playback.

**What was added:**
- `run_real_io_speak_trial(...)` in `src/cora_infans/live_speak_smoke.py`: a bounded,
  gated real-hardware SPEAK smoke. It consults `run_speaking_readiness_gate()` and
  aborts with `verdict=blocked_gates_not_promoted` (no I/O) unless promoted. With a
  promoted gate it runs injected `playback_fn`/`capture_fn`, reads the captured PCM
  back, and estimates roundtrip latency via `estimate_playback_latency_ms`.
- Real primitives: `play_wav_to_device` (ffmpeg audiotoolbox to an explicit device
  index) and `capture_to_wav` (ffmpeg avfoundation, 2 fused channels, bounded
  duration). Both fail loud (no silent fallback). `play_wav_to_device` returns a
  Popen the caller awaits.
- CLI: `--speak-live-smoke` gains `--real-io`, `--input-device` (default `:0`),
  `--output-device-index` (default `2`). `--real-io` and `--fake-io` are mutually
  exclusive. Default (no flag) is unchanged: blocked / fake-I/O only.
- Shared efference prep (`_prepare_efference`) refactors the existing helper so the
  fake and real paths synthesize identical output.

**Safety contract honored:**
- No `--real-io` ⇒ no hardware. Existing blocked/fake behavior untouched.
- `--real-io` without a promoted readiness gate refuses before any speaker/mic use.
- Bounded: fixed `duration_ms` (default 300) and `max_capture_ms` (default 1500),
  no loop, `tag_only` contamination policy (no signal-level subtraction claim).
- Honest channel reporting: capture pinned to 2 fused channels (no 4-mic fabrication).
- `live_playback_allowed` stays `False` in the report — free/looped playback is NOT
  authorized; only this bounded tag-only smoke runs.

**Tests (TDD):** red tests for gate-refusal (no I/O) and promoted-path with injected
fakes (real_hardware=true, finite latency, ordered frame tags); both green. No real
hardware is exercised in CI.

**Honesty boundary:** the real slice is a bounded validation harness, not conversational
TTS. It demonstrates the engineered efference→playback→capture→latency-tag mechanism
step by step; it does not claim biological equivalence or room-calibrated suppression.

**Validation:** full suite: `pytest tests -q` → 185 passed. CLI `--fake-io` path still
passes; usage documents `--real-io`. A single real-hardware run is deferred to an
explicit user request (step 7 of the plan).

### Real-hardware run (2026-07-09, user-approved)

Executed one bounded `--real-io` smoke (vowel `a`, 300 ms output, 1500 ms capture,
output device index 2, input `:0`):

- Readiness gate: `promote_simulation_only` → I/O authorized to proceed.
- `verdict=real_io_complete`, `real_hardware=true`, `live_playback=true`,
  `live_playback_allowed=false`, `usable=true`, `error=None`.
- Playback: ffmpeg audiotoolbox device 2, `returncode=0`.
- Capture: ffmpeg avfoundation `:0`, 2 fused channels, completed.
- Artifacts written: `efference_*.wav` (28.8 KB) and `captured_*.wav` (288 KB ≈
  1500 ms @ 48 kHz stereo).
- `latency_aligned_ms=-1.125`: a weak estimate. The captured vowel is mostly room
  tone (mic distance/angle), so the XCORR found only a small lag. This is honest:
  single-trial latency is not yet well-determined; per the M8-F doc we do NOT claim
  accurate alignment until it is repeatable across trials.

**Fix during the live run:** the first real attempt crashed on JSON serialization
(`Popen` not serializable) because `play_wav_to_device` returned the process handle.
Fixed by awaiting internally and returning a serializable summary (`returncode`);
the CLI `_real_playback` no longer awaits. Suite still 185 passed.

**Honesty note:** this demonstrates the engineered efference→playback→capture→tag
mechanism on real hardware. It is a bounded `tag_only` smoke, not conversational
TTS and not a claim of room-calibrated self-subtraction.

## M8-infans repeatability harness — multi-trial latency spread

**Decision (2026-07-09, user-approved):** before any signal-level self-subtraction
we must prove the playback→capture latency is repeatable. Added a multi-trial
harness so a single noisy trial cannot be promoted to a stable alignment.

**What was added:**
- `run_real_io_speak_trials(...)` in `src/cora_infans/live_speak_smoke.py`: runs
  N bounded real-I/O trials, collects per-trial latencies, and reports
  `latency_stats` (n/mean/std/min/max) plus a `latency_stable` flag. Stable is
  defined honestly: ≥2 trials AND std ≤ `latency_jitter_ms` (default 2 ms). It
  delegates each trial to `run_real_io_speak_trial`, so hardware behavior is
  identical to a single run.
- `run_real_io_speak_trial` gained a `latency_estimator` seam (defaults to the
  XCORR stub) so tests inject a deterministic estimator.
- CLI: `--speak-live-smoke --real-io` gains `--trials N` (N>1 routes to the
  trials harness). Usage updated.

**Tests (TDD):** red tests assert (a) a deterministic injected estimator yields
zero spread and `latency_stable=True`, and (b) a varying estimator (>2 ms spread)
yields `latency_stable=False`. Both green.

**Real-hardware run (4 trials, vowel `a`, 300 ms / 1500 ms):**
- 4/4 trials completed; readiness gate `promote_simulation_only`.
- `latency_stats = {n:4, mean:-1.5, std:7.476, min:-10.417, max:6.479}`.
- `latency_stable=False` — correctly NOT claimed. The captured vowel is mostly
  room tone at the current mic distance/angle, so the XCORR cannot lock a
  consistent delay. This is exactly the honest outcome the harness exists to
  produce; it blocks any premature subtraction claim.

**Honesty boundary:** latency is NOT yet stable, so no signal-level self-
subtraction is authorized. Next step (if pursued) is to improve the capture
condition (mic proximity/orientation, gain, or a fixed reference tone) and repeat
until `latency_stable=True`, only then consider subtraction.

**Validation:** full suite: `pytest tests -q` → 187 passed. CLI `--fake-io` and
`--real-io --trials 1` paths untouched in behavior; `--trials N` routes correctly.

## M8-infans stimulus/volume variants — latency sweep

**Decision (2026-07-09, user-approved):** run tone + chirp probes at two volumes to
find which variant stabilizes playback→capture latency. Higher gain and a swept
chirp were hypothesized to help the XCORR lock a consistent delay.

**What was added:**
- `synthesize_tone` / `synthesize_chirp` in `io/motor_plan.py`: pure calibration
  probes (no biological claim), normalized to [-1,1] with short ramps.
- `play_wav_to_device` gained `volume_db` (forwarded to ffmpeg `volume=…dB`).
- A `StimulusSpec` dataclass + `stimulus` param threaded through
  `run_live_speak_smoke_trial`, `run_real_io_speak_trial`, `run_real_io_speak_trials`.
- CLI `--stimulus {vowel|tone|chirp}`, `--volume-db`, `--tone-hz`, `--chirp-f0`,
  `--chirp-f1`. Usage updated.

**Tests (TDD):** chirp recovers a known delay deterministically at clean SNR;
tone/chirp are bounded unit waveforms; `volume_db` is forwarded to ffmpeg. Suite
now 190 passed.

**Real-hardware sweep (4 trials each, vowel-free, device 2 / input `:0`, 300/1500 ms):**

| variant | trials_ok | mean ms | std ms | min | max | stable |
|---|---|---|---|---|---|---|
| chirp @12dB | 4/4 | 0.48 | 6.57 | -9.19 | 7.48 | False |
| chirp @24dB | 4/4 | 3.89 | 2.93 | -0.17 | 6.98 | False |
| tone @12dB  | 4/4 | 3.79 | 8.30 | -10.35 | 10.40 | False |
| tone @24dB  | 4/4 | -2.23 | 4.97 | -8.94 | 4.10 | False |

**Honest read:** no variant reaches `latency_stable=True` (all std ≫ 2 ms). Clear
trend: louder stimulus tightens spread (chirp 6.57→2.93; tone 8.30→4.97), and the
chirp beats the tone at equal gain. The residual ~3 ms spread is dominated by the
room/reverb path, not the probe. Volume and chirp help, but neither stabilizes
latency at the current mic distance/angle. The honest next lever is mic
proximity/orientation (or a fixed reference-tone burst), not more gain.

**Honesty boundary:** still `tag_only`. Latency is not yet stable, so no
signal-level self-subtraction is authorized. This sweep is a measurement of the
acoustic path, not a claim of alignment fidelity.

**Validation:** full suite: `pytest tests -q` → 190 passed.

## M8-infans causal-window latency fix (from Corollary prior art)

**Decision (2026-07-09, user-approved):** apply the M3/Corollary technique — restrict
the cross-correlation lag search to a causal window around the *known* playback onset
(t=0) instead of a free ±500-sample search. This rejects room-reverb false peaks.

**Investigation:** `corollary/docs/m3_streaming_config_schema.md` and
`packages/thalamus/attributor.py` showed Corollary never free-runs XCORR; it computes
`rel_pos = observed_onset − expected_onset`, correlates only the aligned slice, and
scores timing separately as `causal_score = max(0, 1 − onset_error/causal_tolerance_ms)`
(`causal_tolerance_ms=175ms`). Dry PCM also matched poorly vs a room-colored FIR
reference (self_score 0.09→0.38, `m3_start.md:392`). Step A implements the causal
window (the highest-confidence lever); room-colored reference (step B) remains open.

**What changed:**
- `estimate_playback_latency_ms` gained `expected_onset_ms` (default 0) and
  `causal_window_ms` (default 20 ms; `None` = legacy free search). The lag search is
  restricted to `±causal_window_ms` around the expected onset.
- `run_real_io_speak_trial` / `run_real_io_speak_trials` accept `causal_window_ms`
  (default 20 ms) and pass it to the default estimator. Injected estimators are used
  as-is. `causal_window_ms` is reported in the trials summary.
- CLI `--causal-window-ms F` (default 20). Usage updated.

**Tests (TDD):** under weak playback in strong noise, the windowed estimator's std is
far below the free search and ≤6 ms at an 8 ms window; chirp recovery exact at clean
SNR; volume forwarded. Suite now 191 passed.

**Real-hardware sweep (4" proximity, max speaker volume, device 2 / input `:0`):**

4-variant sweep at cw=8 ms (vs no-window prior run):

| variant | prior std | cw=8 std |
|---|---|---|
| chirp @24dB | 3.38 | 2.92 |
| chirp @12dB | 5.37 | 4.15 |
| tone @24dB | 5.25 | 3.47 |
| tone @12dB | 7.68 | 5.41 |

Tightening further on the best variant (chirp @24dB, 6 trials):
- cw=4 ms → `latency_stable=True`, mean 0.24 ms, std 1.74 ms.
- cw=2 ms → `latency_stable=True`, mean −0.58 ms, std 0.85 ms.

**Result:** the playback→capture loopback latency is now STABLE and measured
(~0 ms at 4" near-field — essentially the electronic buffer plus a negligible acoustic
path) when using a chirp probe at high volume with a tight causal window. `latency_stable`
crossed `True` for the first time. The dominant error was reverb false-peaks in the free
lag search; the causal window removes them.

**Honesty note:** this is a real, repeatable measurement of CORA's own efference
loopback. It is the anchor for frame tagging and (future) self-subtraction. Still
`tag_only`; no subtraction yet. Room-colored reference (step B) is the documented next
improvement if even tighter stability is wanted, but is no longer blocking.

**Validation:** full suite: `pytest tests -q` → 191 passed.

## M8-infans frame tags anchored to stable measured latency

**Decision (2026-07-09, user-approved):** wire the stable measured loopback latency
into the frame tags so tagging (and future self-subtraction) anchors to the
repeatable value, not a single noisy trial.

**What changed:**
- `run_real_io_speak_trials` now publishes `latency_aligned_ms` (the stable mean)
  and `canonical_frame_tags` — a frame-tag set built from the stable latency. It is
  emitted ONLY when `latency_stable=True`; otherwise both are `None` (honest: no
  canonical latency exists yet). Each canonical tag carries `latency_source="stable_mean"`
  and `latency_stable=True`.
- Single-trial reports still tag with their own per-trial measurement (used when not
  running the multi-trial harness).

**Tests (TDD):** the stable-variant trials test asserts `canonical_frame_tags` is a
4-tag set using the stable mean as `latency_aligned_ms` with `latency_source=stable_mean`.
Suite still 191 passed.

**Real-hardware check (chirp @24dB, cw=2ms, 6 trials):** `latency_stable=True`,
`latency_aligned_ms=0.844`, `canonical_frame_tags` present with all four states
(READY_TO_SPEAK / SPEAKING / POST_SPEAK_SUPPRESSION / RECOVERING) tagged at the
measured 0.844 ms. A second run at cw=4ms came back `latency_stable=False` and
correctly omitted `canonical_frame_tags` — the gate holds when the path varies.

**Honesty boundary:** tags are `tag_only`; no subtraction. The canonical latency is
now a real, repeatable anchor available to downstream consumers (the future
imitation loop / self-subtraction) when they appear.

**Validation:** full suite: `pytest tests -q` → 191 passed.

## M8-infans near-field preset + latency calibration artifact (1 + 2)

**Decision (2026-07-09, user-approved "proceed with 1 + 2"):** (1) make the
proven-stable near-field config the default path so `--real-io --trials N` can
actually cross `latency_stable` without manual tuning; (2) persist the measured
stable latency as a calibration artifact for downstream consumers.

**What changed:**
- `--near-field` CLI preset: chirp @24 dB + `causal_window_ms=4`. Explicit flags
  override the preset (preset only fills defaults). Validated 2026-07-09 at 4"
  speaker proximity with max volume.
- `--save-calibration PATH`: writes `calibration_path` into the trials report and
  persists a JSON artifact (kind `playback_capture_loopback_latency`) carrying the
  latency, std, stimulus, volume, window, sample rate, trials, and a UTC timestamp.
  Only written when `latency_stable=True` (honest: never a noisy estimate).
- `save_latency_calibration()` helper + `volume_db`/`save_calibration` params on
  `run_real_io_speak_trials`. Single-trial path does not save (no std).
- Usage string updated.

**Tests (TDD):** `save_latency_calibration` writes an auditable artifact (unit);
the stable-trials test now also asserts `calibration_path` is written and the JSON
contains the stable mean + `tag_only` policy. Suite now 192 passed.

**Real-hardware check:** `--near-field` resolved to chirp@24/cw=4 → 0.917 ms in one
trial; 6-trial run `latency_stable=True` (mean −2.04 ms, std 1.52 ms) with
`canonical_frame_tags` present and `/tmp/cora_nf/latency.json` persisted with full
conditions. The default `causal_window_ms=20` (without `--near-field`) is preserved
for general use; the preset is the validated-fast path.

**Honesty boundary:** tags still `tag_only`; no subtraction. The calibration artifact
is a first-class anchor the future imitation loop / self-subtraction can read — but
only ever for a genuine, repeatable latency.

**Validation:** full suite: `pytest tests -q` → 192 passed.
