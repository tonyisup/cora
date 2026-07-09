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

## Next step

M8-infans-E: integrate the speaking-policy gate with the self-forward-model gate
so a simulated `speak` decision requires both policy promotion and self-prediction
promotion. The output should still say `live_playback_allowed=false`.
