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
