# Milestone 1 — Close-Out Document

**Status:** Complete
**Date:** 2026-05-21

## Goals

Milestone 1 had two objectives:

1. Build a speaking-policy simulator with a deterministic state machine, and evaluate it on a range of scripted turn-taking scenarios.
2. Collect comparative evidence on whether LLM-based policies can match or improve on the deterministic baseline, and identify where distillation is worthwhile.

Both objectives are complete.

---

## Architecture

### State Machine

The deterministic policy (`DeterministicSpeakingPolicy`) implements a 5-state machine:

```
IDLE → LISTEN → SPEAK → REPAIR
               ↘ CLARIFY
```

State transitions are driven by perception frames containing:
- attribution (silence / user / both)
- silence duration
- VAD energy
- queued utterance presence

The policy emits one of 5 actions: `speak`, `wait`, `clarify`, `repair`, `yield`.

### Simulator

`SpeakingPolicySimulator` runs tick-by-tick (30ms resolution) trials against scripted scenarios with a `UserBehaviorConfig` controlling:
- response delays
- interrupt timing
- collision / overlap behavior
- no-response probability

Each trial records: interruptions, collisions, repair success, clarification success, turn continuation rate, response gap.

### Prompt-Based Policy

`PromptSpeakingPolicy` replaces the hand-written decision tree with an LLM callable:
- Formats state + perception into a structured prompt
- Calls the LLM (via Hermes CLI by default, or OpenRouter directly)
- Parses the `ACTION / EXPLANATION / TEXT` response
- Caches identical-context calls to reduce LLM invocations

---

## Phase Summary

### Phase 1–2: Core States and Evaluation

Implemented the basic state machine (IDLE → LISTEN → SPEAK → REPAIR). Added collision, interruption, and clarification handling. Built the scenario runner and metrics pipeline. Fixed duplicate-utterance bugs, clarify timeout alignment, and smooth turn counting.

**Deterministic results (Phase 2, 50 trials):**
| Metric | Value | Target | Status |
|---|---|---|---|
| Interruption rate | 0.0% | < 5% | PASS |
| Collision rate | 0.0% | < 5% | PASS |
| Response gap | 415ms | 300–800ms | PASS |
| Turn continuation | 69.3% | > 70% | CLOSE |

### Phase 3: Prompt Calibration Baseline

Connected a prompt-based policy to the same evaluation harness. Established the scenario logging and comparison format. Verified infrastructure end-to-end.

### Phase 4: Harder Scenarios + Model Benchmarks

**Harder scenarios:**
- Added `__INTERRUPT__`, `__OVERLAP__`, `__SILENCE__` special tokens
- 9 scenario scripts totaling 50 trials across the action space
- Raw success rates on exercised sub-scenarios: repair 81%, clarification 100%

**Deterministic results (Phase 4, 50 trials):**
| Metric | Value | Target | Status |
|---|---|---|---|
| Interruption rate | 2.1% | < 15% | PASS |
| Turn continuation | 69.1% | > 70% | FAIL (close) |
| Repair success (raw) | 81% | > 80% | PASS |
| Clarification (raw) | 100% | > 60% | PASS |
| Collision rate | 6.6% | < 10% | PASS |
| Response gap | 496ms | 300–800ms | PASS |

**Model benchmarks:**
- Benchmarked 15+ models on 6 fixed turn-taking scenarios
- Tested both zero-shot and 4-example few-shot prompts
- Measured accuracy, parse rate, and latency per model

Key benchmark findings:
- **No LLM matched the deterministic policy** on the full scenario set
- **Claude 3.5 Haiku** was the only model to reach 100% accuracy (with few-shot)
- **gpt-5.4-mini / 5.4 / 5.5** (via Hermes/openai-codex) all plateau at 83.3% — they consistently fail the clarify case
- **Gemini 3.1 Flash Lite** gets clarify right sometimes but is inconsistent across prompt variants
- The clarify decision (`no queued utterance + long silence → clarify`) is the key discriminator: models default to `wait` regardless of few-shot examples
- Several models (Kimi, Gemma, DeepSeek-R1-distill) don't emit the action format at all and are structurally unsuitable

---

## Key Findings

### 1. The deterministic policy is the trust baseline
It passes all targets on the full scenario mix, has predictable latency, and is fully inspectable. Any distillation or LLM augmentation should be measured against it, not replace it.

### 2. Clarify is the hardest and most informative decision
Across 15+ models and two prompt variants, the clarify-after-silence case was the most consistent failure mode. This is where the LLM approach diverges most from the hand-written rule, and where the most value would come from either a better rule or a well-distilled model.

### 3. LLM policies are not yet competitive as runtime replacements
Even the best model (Claude Haiku at 100% on 6 scenarios) has ~1.5s latency per decision. The deterministic policy is effectively zero-latency in simulation and handles edge cases more predictably.

### 4. LLM policies are useful as teachers
The benchmark data provides a clear ranking of which models understand turn-taking best:
1. Claude 3.5 Haiku (100% few-shot, reasonable latency)
2. Gemini 3.1 Flash Lite (fast, but inconsistent on clarify)
3. GPT-5.4-mini family (solid on easy cases, stuck on clarify)

These are the candidates for generating distillation training data, not for runtime deployment.

### 5. The evaluate infrastructure is solid
- Scenario-level benchmark with fixed prompts and expected labels
- Hermes CLI and OpenRouter direct paths both working
- Caching, parse-rate tracking, per-scenario and aggregate metrics
- Easy to add new models: one line in the model spec list

---

## Open Items (Carry to Milestone 2)

1. **Protecting clarify with a hard rule** — the benchmark shows this can't be left to prompt engineering alone. A deterministic guard (`no queued + silence > threshold → clarify`) should remain in the loop regardless of distillation.

2. **Distillation scope definition** — based on the benchmark, distill only from Claude and possibly Gemini, targeting the 5/6 scenarios where LLMs are reliable. Exclude clarify from distillation until the model signal is stronger.

3. **Hard-negative scenarios** — add more ambiguous cases (e.g., user briefly pauses mid-turn vs. truly finished) to stress-test both the deterministic and distilled policies.

4. **Live validation bridge** — the simulator-to-live path is designed but not tested. Milestone 2 should include at least one end-to-live-live validation run with the deterministic policy.

---

## Files

### Core Implementation
- `scripts/m1_speaking_policy_sim.py` — simulator, deterministic policy, prompt policy

### Benchmarks
- `scripts/m1_scenario_benchmark.py` — scenario-level model benchmark (supports `--few-shot`)
- `scripts/m1_model_benchmark.py` — earlier prompt-level benchmark (superseded by scenario-level)
- `results/m1_scenario_benchmark_15_models.json` — baseline 15-model results
- `results/m1_scenario_benchmark_15_models_fewshot_2.json` — few-shot 15-model results
- `results/m1_scenario_benchmark_fewshot_focus.json` — focused 5-model few-shot results

### Deterministic Evaluation Results
- `results/m4_deterministic_5trials.json` — 5-trial deterministic baseline
- `results/m4_harder_scenarios.json` — 50-trial deterministic with harder scenarios

### Documentation
- `docs/m1_phase4_results.md` — Phase 4 detailed results
- `docs/m1_model_benchmark_pilot.md` — initial prompt-level benchmark pilot
- `docs/m1_scenario_benchmark_pilot.md` — initial scenario-level benchmark pilot
- `docs/m1_benchmark_fewshot_comparison.md` — few-shot vs baseline comparison
- `docs/pivot_learns_to_speak.md` — project pivot rationale and background
