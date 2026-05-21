# M3 — LLM User Agent Expansion Implementation Plan

> **For Hermes:** Use `subagent-driven-development` if you execute this plan. Keep M2 as the baseline and do not remove the deterministic guard rails.

**Goal:** Replace the scripted user with an LLM user agent to generate more diverse, realistic conversations, then retrain and evaluate the distilled speaking policy against that broader interaction distribution.

**Architecture:**
M2 proved the speaking policy can be distilled into a small learned model, but the training distribution was still too narrow and too synthetic. M3 expands the environment, not the action space: the user side becomes an LLM-driven conversational agent while the speaking policy remains the same small action set (`speak`, `wait`, `clarify`, `repair`, `yield`) with deterministic safety rails. The primary output of M3 is a stronger, more diverse offline dataset and a retrained policy that generalizes beyond the scripted scenarios.

**Tech Stack:**
- Existing M1/M2 Python simulator and data pipeline
- local-omlx/gemma-4-26b-a4b-it-4bit for LLM calls, Hermes/OpenAI Codex or OpenRouter fallbacks where needed
- JSONL dataset logging
- PyTorch MLP policy model
- Existing scenario benchmark harnesses

---

## What M3 is and is not

M3 is:
- an LLM-user simulator upgrade
- a data distribution expansion milestone
- a retraining/evaluation pass for the distilled speaking policy
- a robustness check for turn-taking, clarification, repair, and interruption handling

M3 is not:
- a new action space
- RL/PPO
- raw waveform generation
- a removal of the deterministic guardrails

The point is to make the conversation environment more realistic while preserving the safety envelope that M2 established.

---

## Success criteria

M3 is successful if:
1. The LLM user agent produces more diverse interaction traces than the scripted user.
2. The distilled policy remains stable or improves on the M2 scenario suite.
3. The policy retains high accuracy on interruption/yield and repair.
4. The new dataset contains materially more varied clarification, correction, and overlap cases.
5. The evaluation remains fully reproducible and logged to JSONL.

Suggested pass criteria:
- raw distilled scenario accuracy: >= M2 raw accuracy
- guarded scenario accuracy: 100%
- clarify accuracy on validation set: does not regress below M2 v3 materially
- yield accuracy on interruption/overlap: remains 100% with guard, and ideally improves raw
- dataset diversity: measurable increase in unique turn patterns / prompt variants / repair situations

---

## Milestone structure

### Task 1: Define the LLM user-agent contract

**Objective:** Specify exactly what the user agent is allowed to see, say, and return.

**Files:**
- Create: `docs/m3_user_agent_contract.md`
- Modify: `scripts/m2_collect_data.py` or successor pipeline if the interface is embedded there

**Deliverable:**
A small contract that defines:
- input state to the user agent
- output format from the user agent
- allowed personality / behavior knobs
- safety boundaries (no self-harm, no illicit content, no uncontrolled escalation)

**Minimum fields to define:**
- conversation history slice
- current speaker / turn index
- last agent utterance
- last user utterance
- task goal / scenario label
- whether the agent should answer, ask a question, or stay silent
- output text plus optional metadata such as confidence or stance

**Verification:**
- The contract can be represented as JSON
- A sample input/output pair is documented
- No ambiguity about who speaks next or how turns are counted

---

### Task 2: Build the LLM user-agent generator

**Objective:** Replace the scripted user with an LLM-driven responder that can produce varied conversational turns from the same scenario seed.

**Files:**
- Create: `scripts/m3_user_agent.py`
- Create: `scripts/m3_generate_dialogues.py`
- Modify: any simulator entrypoint that currently instantiates the scripted user

**Behavior to implement:**
- generate a user response given a prompt/state packet
- optionally emit silence / no-response when the scenario calls for it
- preserve scenario labels and replayability with a seed
- support both direct and fallback model configs

**Implementation notes:**
- Keep the prompt compact and scenario-conditioned
- Prefer deterministic decoding where possible for repeatability
- Record raw model output and latency per user turn
- Do not let the user agent access the policy label directly unless the scenario requires it

**Verification:**
- Given the same seed and prompt, a run can be reproduced or at least deterministically approximated
- JSONL traces include raw user outputs, timing, and scenario IDs
- User responses vary across scenarios and are not just template echoes

---

### Task 3: Add dialogue generation logging

**Objective:** Store complete offline traces so the training pipeline can ingest them later.

**Files:**
- Create: `results/m3_dialogues/*.jsonl` or a timestamped equivalent
- Modify: `scripts/m3_generate_dialogues.py`

**Required logging fields:**
- scenario name / prompt type
- turn index
- user-agent raw response
- assistant policy decision
- assistant utterance text when present
- timestamps / latency
- parsed actions
- parse failures
- final outcome tags

**Verification:**
- A single run produces valid JSONL
- The JSONL can be consumed by the existing training pipeline or a thin adapter
- At least one dialogue includes a clarification, one includes a repair, and one includes an interruption/overlap-like turn pattern

---

### Task 4: Expand the scenario bank

**Objective:** Add more diverse, realistic scenarios that the LLM user agent can drive.

**Files:**
- Modify: `scripts/m2_collect_data.py` or the M3 scenario source of truth
- Create or update: `docs/m3_scenarios.md`

**Scenario categories to add:**
- user gives partial information and later corrects themselves
- user changes topic mid-response
- user asks a follow-up after an answer
- user does not answer and returns later
- user interrupts the assistant and then apologizes
- user accepts a repair, rejects a repair, or asks for rewording
- user provides ambiguous feedback requiring clarification

**Verification:**
- The scenario bank includes cases that are not trivially solved by the existing M2 policy
- Each scenario has a clear expected policy outcome or labeled uncertainty bucket
- Scenario definitions are concise enough to replay offline

---

### Task 5: Retrain the speaking policy on M3 traces

**Objective:** Train the distilled policy on the expanded conversation distribution.

**Files:**
- Modify: `scripts/m2_train.py` or create `scripts/m3_train.py`
- Modify: data loading if the new dialogue schema needs an adapter

**Training strategy:**
- Keep the compact MLP policy architecture unless the new traces justify a change
- Continue class-weighting and oversampling for rare actions
- Keep deterministic safety rails in the runtime wrapper
- Add any new labels only if they are absolutely necessary and well-measured

**Verification:**
- Training runs successfully on the new dialogue traces
- Validation metrics are reported per action
- The raw and guarded scenario suites are both evaluated after training

---

### Task 6: Evaluate against M2 and baseline

**Objective:** Determine whether the LLM user agent actually improved the policy learning problem.

**Files:**
- Modify: `scripts/m2_eval.py` or create `scripts/m3_eval.py`
- Update: `docs/m3_results.md` or similar evaluation report

**Evaluation dimensions:**
- scenario-suite accuracy
- per-action accuracy
- clarify accuracy
- yield-on-interrupt accuracy
- repair correctness
- raw vs guarded performance
- dataset diversity compared to M2
- average latency for user-agent generation

**Required comparison:**
- M2 distilled v3 vs M3 retrained policy
- raw vs guarded for both
- scenario-by-scenario diffs for all misses

**Verification:**
- A table shows where M3 improved or regressed relative to M2
- If M3 does not improve the policy, the report explains why
- The evaluation preserves raw outputs and timing so the comparison remains apples-to-apples

---

### Task 7: Decide whether to continue Track 1 or pivot to Track 2

**Objective:** Use the M3 evidence to decide whether to keep pushing the practical speaking-policy track or move more energy into the sound-first motor-learning track.

**Files:**
- Update: `docs/pivot_learns_to_speak.md` or a dedicated decision memo
- Update: `docs/m3_results.md`

**Decision rules:**
- If M3 meaningfully improves conversational diversity and policy robustness, continue Track 1
- If M3 adds cost but little behavioral gain, freeze Track 1 and pivot more toward the acoustic learning lane
- If the LLM user agent introduces too much instability, keep the scripted offline environment as the canonical training source

**Verification:**
- The report makes a clear recommendation
- The next milestone is explicit: continue Track 1, harden M3, or pivot to Track 2

---

## Proposed file map

Likely new files:
- `scripts/m3_user_agent.py`
- `scripts/m3_generate_dialogues.py`
- `scripts/m3_train.py` or targeted modifications to `scripts/m2_train.py`
- `scripts/m3_eval.py` or targeted modifications to `scripts/m2_eval.py`
- `docs/m3_user_agent_contract.md`
- `docs/m3_scenarios.md`
- `docs/m3_results.md`

Likely reused files:
- `scripts/m2_collect_data.py`
- `scripts/m2_train.py`
- `scripts/m2_eval.py`
- `scripts/m2_guarded_policy.py`
- `results/m2_*`

---

## Risks and mitigations

### Risk: the LLM user agent is too expensive or too slow
Mitigation:
- sample fewer scenarios per run
- cache repeated user-agent prompts
- use smaller/faster models for data generation where acceptable

### Risk: the new data is noisier than M2 and hurts training
Mitigation:
- preserve raw traces
- label and filter weak traces
- keep deterministic M2 data in the mixture

### Risk: the policy starts chasing user-agent artifacts
Mitigation:
- keep the scripted scenario bank as a validation anchor
- compare on both scripted and LLM-generated distributions

### Risk: the action space expands before the environment stabilizes
Mitigation:
- do not add new actions until M3 proves they are needed and measurable

---

## Recommended order of execution

1. Define the user-agent contract
2. Build the LLM user-agent generator
3. Log dialogue traces
4. Expand the scenario bank
5. Retrain the distilled policy
6. Evaluate against M2
7. Decide whether to continue Track 1 or pivot

---

## Bottom line

M3 is the first real test of whether the speaking policy can survive a less synthetic conversation distribution.

M2 answered: can we distill the policy?

M3 answers: does that policy still hold up when the user is no longer scripted?
