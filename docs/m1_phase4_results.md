# M1 Speaking Policy — Phase 4 Results

## Summary

Phase 4 focused on two goals: (1) adding harder scripted scenarios that exercise
the full action space, and (2) connecting a real LLM callable for prompt-based
policy evaluation.

## Step 1: Harder Scripted Scenarios — COMPLETE

Added 4 new scenario scripts (9 total) with special response tokens:
- `__INTERRUPT__` — user interrupts agent mid-utterance
- `__OVERLAP__` — user starts speaking simultaneously (collision)
- `__SILENCE__` — user doesn't respond (triggers clarify)

Enhanced `UserBehaviorConfig`:
- Lowered interrupt threshold from 3000ms to 1500ms
- Added `interrupt_delay_mean_ms` (800ms) for realistic interrupt timing
- Added `no_response_probability` (8%) for stochastic silence

Bug fixes:
- `on_yield()` now preserves REPAIR state (was overwriting to LISTEN)
- REPAIR action emitted on any collision (not just after 200ms overlap)
- `on_speech_start`/`on_speech_end` called for REPAIR and CLARIFY actions
- Repair/clarify success tracking when user responds
- `_will_not_respond` flag preserved across clarify utterance

### Results (50 trials, seed=42, deterministic policy):

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Interruption rate | 2.1% | < 15% | PASS |
| Turn continuation | 69.1% | > 70% | FAIL |
| Repair success | 11.6% (raw: 81%) | > 80% | FAIL* |
| Clarification efficiency | 24% (raw: 100%) | > 60% | FAIL* |
| Collision rate | 6.6% | < 10% | PASS |
| Response gap | 496ms | 300-800ms | PASS |

*Averaged metrics diluted by easy trials with no repairs/clarifications.
Raw success rates on exercised scenarios: 81% repair, 100% clarification.

### Scenario-conditioned breakdown:
- Repairs: 26/32 = 81% success (on trials with collisions)
- Clarifications: 18/18 = 100% success (on trials with silence)
- Interruptions: 6 across 50 trials
- Collisions: 32 across 50 trials

## Step 2: LLM Callable — INFRASTRUCTURE COMPLETE, EVALUATION IMPRACTICAL

Built `hermes_llm_callable()` function that:
- Calls `hermes chat -q <prompt> --quiet` via subprocess
- Parses response to extract ACTION/EXPLANATION/TEXT
- Caches responses to avoid redundant calls for identical contexts
- Falls back to deterministic policy on timeout/error

Added `--llm` CLI flag to enable the hermes callable for prompt policy.

### Performance Issue

The hermes CLI takes ~8 seconds per call. Even with caching (~20-30 unique
contexts per trial), a single trial takes 2.5-4 minutes. For 50 trials, this
would be 125-200 minutes — impractical for iterative evaluation.

A single trial was tested successfully (completed with LLM decisions), confirming
the infrastructure works. But full evaluation requires either:
1. A faster LLM endpoint (local model, lower-latency API)
2. Fewer trials (5-10 for calibration purposes)
3. Pre-computed LLM responses cached to disk

### Recommendation

The deterministic baseline is solid and the scenario coverage is good. The LLM
callable infrastructure is ready. For meaningful comparison:
- Run 5-10 trials with LLM (accepting ~20-40 min runtime)
- Or integrate a faster LLM endpoint
- Report both scenario-conditioned and aggregate metrics

## Files Modified

- `scripts/m1_speaking_policy_sim.py` — harder scenarios, hermes callable, bug fixes
- `results/m4_deterministic_5trials.json` — deterministic baseline (5 trials)
- `results/m4_harder_scenarios.json` — deterministic baseline (50 trials)
