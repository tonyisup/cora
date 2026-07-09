# Phase 2: Live Self-Subtraction (forward-model learning cycle)

**For Hermes:** Present-first per user rule. Write this plan, then implement task-by-task with TDD. Do NOT start coding until the plan is written AND the user says proceed (they said "proceed with writing the phase 2 plan" — writing only; execution awaits their go).

**Goal:** Turn the Phase 1 measurement-only loop into a true forward-model learning cycle: reuse `io/self_suppression.py`'s `predict_self_pcm` + `spectral_subtract` with the **measured device-3 latency anchor** and the **real efference + real captured corollary mixture**, so Cora's own played audio is subtracted from the mic capture, isolating the human's continuing speech (`world_residual`). Report SVSR + world-residual correlation as evidence. Still gated, bounded, no free/looped playback.

**Anchor (user-confirmed):** `/tmp/cora_nf/latency_dev3.json` → `latency_aligned_ms = -0.62`, `std = 1.03`. Fed to `predict_self_pcm(delay_ms=...)` for the imitation path. Playback + capture use device 3 (imitation) / `:0` (mic). World/test stimulus stays device 2.

## Grounded API (from `io/self_suppression.py`, read this session)
- `predict_self_pcm(efference_pcm, *, sample_rate=16000, delay_ms=10.0, attenuation=0.65, echo_decay=0.18) -> float32` — toy forward model: `out[delay:] += attenuation*eff[:len-delay]` plus a fixed 15 ms echo (`echo_delay = max(1, round(15*sr/1000))`, NOT a param) at gain `attenuation*echo_decay`.
- `spectral_subtract(observed_pcm, predicted_self_pcm, *, floor=0.0) -> float32` — `obs - pred`, clipped. The actual subtraction op.
- `evaluate_forward_model(eff, obs, *, sample_rate, delay_ms, attenuation, echo_decay, world_pcm=None)` → `{params, observed_rms, predicted_rms, residual_rms, svsr_db, world_residual_correlation?}`.
- `fit_forward_model(eff, obs, *, sample_rate, delay_grid_ms, attenuation_grid, echo_decay_grid, world_pcm=None, svsr_pass_db=20.0)` → best grid setting `{learned_params, svsr_db, residual_rms, verdict}`.
- `compute_svsr_db(observed, residual)`, `rms(pcm)`, `_corr(a,b)`.
- `run_forward_model_promotion_gate(...)` is **simulation-only** (sets `live_playback_allowed=False`); do NOT reuse it for Phase 2 live promotion. Phase 2 needs its own real-data gate.

## Honesty boundaries (hard)
- Bounded + gated: `--real-io` + speaking readiness gate must pass before any playback; no looped/free playback (unchanged from Phase 1).
- `contamination_policy` transitions `tag_only` → `self_subtract_live` **only** when a NEW real-data promotion gate passes (SVSR threshold + world-signal preservation on actual captured audio). Default remains `tag_only`. The subtraction is recorded as secondary evidence even in `tag_only`.
- "Learning" = transparent grid-search fit of delay/attenuation/echo on real efference→observation pairs. No black box, no weight updates, no biological overclaim.
- Sample-rate match is mandatory: efference used for subtraction MUST be at the capture rate (48000), not the 16000 Phase-1 analysis rate. The toy `predict_self_pcm` echo is hard-coded at 15 ms; if real-loopback fit is poor, extend its signature to accept `echo_delay_ms` (flagged as refinement, not assumed).

## Critical integration facts from Phase 1 code
- `_motor_imitation_trial` currently: writes `efference_wav` at `sample_rate=16000` (`write_wav(..., sample_rate=16000)`), plays it via `playback_fn` (device 3, 48000), captures corollary via `capture_fn` (48000), re-analyzes the PLAYED wav for `played_formants_hz`. For Phase 2 the efference PCM handed to `predict_self_pcm` must be the **48000 resynth** (or resampled), aligned to the 48000 captured window.
- Capture window = corollary feedback; its start is anchored by `latency_aligned_ms` (the device-3 anchor). The observed mixture = [self (Cora's played imitation) + world (human continuing speech, if any)].
- `predict_self_pcm` with `delay_ms=-0.62` → `int(round(-0.62*48000/1000)) = int(round(-29.76)) = -30` → `max(0, -30) = 0`. So at sr=48000 the measured negative delay collapses to delay=0. The real loopback is sub-sample/near-zero; we must fit `delay_grid_ms` around 0 (e.g. (−2, 0, 2, 4)) and rely on attenuation/echo to match gain/shape. Flag: negative delay cannot be represented by the toy model (it only shifts predicted forward); if fit demands negative, extend signature or accept delay=0 floor.

---

## Step-by-step plan (TDD, one commit per task)

### Task 1: Fit real forward-model params per trial (no subtraction yet, measurement-only)
**Objective:** Inside `_motor_imitation_trial`, after capture, run `fit_forward_model` on (48000 efference_pcm, 48000 corollary_pcm) with delay grid around the anchor, attenuation grid (0.3–0.9), echo grid (0.05–0.35). Record `forward_model_fit` (learned params + svsr_db + residual_rms + verdict) in the trial dict. No signal is subtracted or routed anywhere yet.
- Files: `io/live_listener.py` (`_motor_imitation_trial`), `tests/test_live_listener.py`.
- Test: fake efference + fake observed (efference delayed/attenuated by known params + a small world sine) → assert fit recovers attenuation within grid resolution and `svsr_db` > 0; world_residual_correlation not None.
- Commit: `feat(phase2/task1): fit real forward-model params per imitation trial (measurement-only)`.

### Task 2: Apply `spectral_subtract` and produce `world_residual` (still not routed)
**Objective:** Use the fitted (or anchor-default) params to call `predict_self_pcm` then `spectral_subtract(observed_corollary, predicted_self)`, yielding `world_residual_pcm` (in-memory ndarray; not played, not saved unless `--out-dir`). Record `svsr_db`, `residual_rms`, `observed_rms`, `world_residual_correlation` (vs the perceived human frame PCM as proxy world). Keep `contamination_policy` = `tag_only`.
- Files: `io/live_listener.py`, `tests/test_live_listener.py`.
- Test: injected efference + observed (self+world) → assert `spectral_subtract` residual RMS < observed RMS and `world_residual_correlation` > 0 (world preserved).
- Commit: `feat(phase2/task2): apply spectral_subtract, emit world_residual metrics (tag_only)`.

### Task 3: Real-data promotion gate + honesty transition
**Objective:** Add `_real_self_subtraction_promotion_gate(residual_rms, observed_rms, svsr_db, world_residual_correlation, *, svsr_pass_db, min_world_corr)` returning `promote`/`hold`. When `promote`, the trial report sets `contamination_policy = "self_subtract_live"`; otherwise stays `tag_only`. This is SEPARATE from the sim-only `run_forward_model_promotion_gate`.
- Files: `io/live_listener.py`, `tests/test_live_listener.py`.
- Test: gate promotes when svsr_db ≥ threshold AND world_corr ≥ min; holds otherwise.
- Commit: `feat(phase2/task3): real-data self-subtraction promotion gate + policy transition`.

### Task 4: CLI wiring — `--with-self-subtraction`
**Objective:** Add `--with-self-subtraction` to `--listen-live` (requires `--real-io` + readiness gate; error/refuse if missing). Pass `do_self_subtraction=True` + anchor to `listen`. Default OFF → Phase 1 tag_only behavior unchanged.
- Files: `cli.py`, `tests/test_cli.py`.
- Test: parse flag; flag without `--real-io` raises/refuses; with both, listener receives `do_self_subtraction=True`.
- Commit: `feat(phase2/task4): --with-self-subtraction CLI flag (gated by --real-io)`.

### Task 5: End-to-end integration test (injected fakes)
**Objective:** Drive `listen(..., with_motor_imitation=True, with_self_subtraction=True, real_io=True, playback_fn/capture_fn/gate_fn injected)`; assert each trial has `world_residual` metrics, `forward_model_fit`, and `contamination_policy` set per gate.
- Files: `tests/test_live_listener.py`.
- Commit: `test(phase2/task5): end-to-end self-subtraction integration test`.

### Task 6: Real-hardware bounded smoke (YOU confirm before running)
**Objective:** Run on device 3 / `:0` with the dev-3 anchor; report svsr_db + world_residual_correlation; bounded; policy per gate.
- Command (after your go):
  `PYTHONPATH=src python3 -m cora_infans --listen-live --with-motor-imitation --with-self-subtraction --with-attention --with-stimulus --real-io --output-device-index 2 --imitation-device-index 3 --volume-db 12 --latency-anchor /tmp/cora_nf/latency_dev3.json`
- Assert: trials carry `forward_model_fit` + `world_residual` metrics; `contamination_policy` ∈ {`tag_only`, `self_subtract_live`}; no free playback.

---

## Files likely to change
- `src/cora_infans/io/live_listener.py` (trial: fit + subtract + gate + policy)
- `src/cora_infans/cli.py` (`--with-self-subtraction`)
- `src/cora_infans/io/self_suppression.py` (possible: add `echo_delay_ms` param to `predict_self_pcm` if real fit demands; NOT assumed)
- `tests/test_live_listener.py`, `tests/test_cli.py`
- `docs/PROJECT_STATUS_2026-07-09.md` (append Phase 2 results)

## Validation
- `pytest tests -q` green throughout (currently 201 passed). New tests per task. Full suite must stay green.
- Real-hardware smoke only after explicit user go (Task 6).
- Anchor artifacts stay in `/tmp` (user: "leave in tmp for now").

## Risks / tradeoffs / open questions
- **Toy model can't represent negative delay:** measured anchor −0.62 ms → delay=0 at 48k. If real loopback needs negative shift, extend `predict_self_pcm` or accept delay=0 floor. Fit grid covers (−2..4) ms to probe.
- **Hard-coded 15 ms echo** in `predict_self_pcm` may not match real device-3→`:0` loopback (likely shorter). If fit svsr is poor, extend signature with `echo_delay_ms`. Out of scope unless Task 1 fit shows it's needed.
- **Sample-rate alignment:** efference for subtraction must be 48000 (capture rate), not the 16000 Phase-1 analysis WAV. Resynth at 48000 or resample.
- **Attenuation/gain mismatch:** sim default 0.65 won't match real loopback gain; Task 1 fit on real data is the correction, not assuming sim defaults.
- **World proxy:** `world_residual_correlation` uses the perceived human frame as proxy world (imperfect — the captured world may differ). Honest caveat: correlation is a sanity signal, not ground truth.
- **No biological claim:** this is transparent grid-search subtraction; we measure SVSR, we do not claim phonetic fidelity or learning.
