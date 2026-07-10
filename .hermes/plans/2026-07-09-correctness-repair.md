# Correctness Repair Plan — 2026-07-09

## Safety boundary

- Treat today's latency anchors and hardware conclusions as invalid.
- Keep `contamination_policy=tag_only`.
- Do not run hardware until deterministic fixes pass and the user sees the new run procedure.
- Do not reuse `promote_simulation_only` as live-I/O authorization.

## Task 1 — Real-I/O coordination and evidence integrity

TDD regressions:
- Capture must be active before playback starts and remain active through playback.
- Non-zero playback exit must fail loudly.
- Silent/low-confidence latency captures must return no estimate.
- Stereo WAV loading must preserve frame/channel shape and produce a correct mono float stream.

Implementation:
- Introduce an explicit asynchronous capture handle/start-wait boundary.
- Record capture-start and playback-onset timestamps.
- Use one channel-aware WAV loader for calibration and self-subtraction.
- Keep unique per-trial artifacts.

## Task 2 — Sample-rate and efference identity

TDD regressions:
- Real 48 kHz listener PCM is analyzed at 48 kHz.
- No fake 16→48 repeat occurs for already-48 kHz PCM.
- Forward fitting uses the exact PCM emitted by playback, not independently re-synthesized PCM.

Implementation:
- Carry sample-rate metadata with retained PCM.
- Prepare one explicit output-rate efference buffer; write/play/model that same buffer.

## Task 3 — Safety and promotion semantics

TDD regressions:
- `live_playback_allowed=False` prevents playback even with `promote_simulation_only`.
- No-playback/no-capture/fitting exceptions cannot yield a successful trial.
- Promotion requires stable calibration plus held-out synchronized evidence.
- Fit/score uses unclipped float residuals.

Implementation:
- Separate simulation readiness from explicit bounded-live authorization.
- Fail loudly and serialize error evidence.
- Keep live policy `tag_only` until held-out synchronized world evidence exists.

## Task 4 — Policy and scheduler state

TDD regressions:
- Negative-reward episodes cannot promote.
- Speak consumes queue/starts speech; yield stops speech and defers output.
- Collision without active agent speech recovers rather than deadlocking.
- Repair clears deferred state.
- Motor and acoustic events use one clock domain and preserve vowel metadata.
- Custom theta/gamma values drive summaries.

Implementation:
- Use independent per-episode acceptance targets.
- Enact all policy actions in the simulator.
- Repair tracker transitions and reward window.
- Thread real frame timestamps into motor scheduling.

## Task 5 — Integration and documentation

- Run focused RED/GREEN cycles per task.
- Run `pytest tests -q`.
- Independent spec and quality review.
- Update project status to retract invalid hardware conclusions and document rerun prerequisites.
- Stop before real hardware rerun for user-visible setup confirmation.
