# Integrated progress hero demo specification

Status: v1 frozen implementation specification. This is a presentation and
bounded integration layer over already-retained results. It does not revise a
VISION claim or promote a stopped research program.

## 1. Demonstrated experience

One run must make the following causal chain visible and audible:

1. receive an opaque 30-dimensional auditory target;
2. begin at the neutral/rest motor prototype;
3. expose the frozen 12-attempt local search and its best-so-far progress;
4. recall the selected prototype for attempts 10--12 without further search;
5. render the selected motor as an exact, SHA-256-identified efference WAV;
6. use one prior replay to bind that exact action to its microphone consequence;
7. verify a repeated self probe, preserve an overlapping world event, and expose
   a wrong-efference negative control;
8. show the retained Stage 6 operating-point measurement with an explicit
   `retained_not_measured_this_session` label; and
9. emit a complete report and a self-contained interactive dashboard whose
   visual states are derived only from that report.

The hero demo is not permitted to animate a successful state that is absent
from its JSON report.

## 2. Source modes

### Project-owner replay

The preferred source is a completed `VOCAL_IMITATION_PROTOCOL.md` session. The
runner must verify the session result, capture manifest, frozen association,
and every teacher/learned waveform hash it consumes. Raw recordings stay
outside Git. The selected vowel is display metadata; action selection is the
already-retained label-blind result.

### Synthetic replay

The hardware-safe fallback runs `DEMO_SPEC.md` v3 cross-configuration learning
and uses only generated waveforms. It exists for deterministic tests, portable
demonstration, and dashboard QA. It is not real-audio evidence.

## 3. Bounded exact-efference panel

Each run selects one target and one distinct learned prototype as the negative
control. It renders five conditions:

1. `self_calibration`: the exact selected efference, used only as the prior
   sensory template;
2. `wrong_reference`: a distinct learned efference, used only for the negative
   control;
3. `self_probe`: an exact repeat of `self_calibration`;
4. `world_only`: the teacher/world waveform without a bound self identity; and
5. `overlap`: the exact self efference and teacher/world waveform concurrently.

Every track uses 16 kHz PCM, a 100 ms alignment chirp, 150 ms pre-stimulus
silence, the bounded stimulus, and 150 ms tail silence. The exact self and world
tracks are peak-limited before mixing. Total physical playback is bounded to
five tracks and less than ten seconds.

The phase- and gain-invariant 12-band spectral shape and frozen H3 threshold
`0.07895991087871712` are reused without tuning. This reuse is an integration
check, not a new H3 promotion. The panel passes only when:

- `self_probe` is below threshold and is suppressed;
- `world_only` is preserved because it has no bound self identity;
- `overlap` is at or above threshold and is preserved;
- the `self_probe` compared with `wrong_reference` is at or above threshold and
  therefore cannot be suppressed by the wrong efference;
- every capture is finite and contains zero clipped samples; and
- all action, track, capture, and report artifacts are SHA-256 identified.

## 4. Execution modes and physical boundary

### Replay mode

Replay mode applies a deterministic bounded room transform to the rendered
tracks. Reports must carry `evidence_class=presentation_fixture`,
`real_hardware=false`, and `live_playback=false`. A replay pass verifies the
runner, scoring, controls, and presentation only.

### Live mode

Live mode is fail-closed and requires all of:

- literal `--authorize-live-io` authorization;
- a passing read-only preflight against `config/operational-point.json`;
- XVF3800 input `:0`, 16 kHz, and two fused host channels;
- self output device 3, Mac mini Speakers, centered beneath the array at 4 cm,
  100% system volume, and frozen `-18 dB` digital playback gain;
- world output device 2, External Headphones, 180 degrees and 50 cm from the
  array, approximately 50% hardware volume, and frozen `-18 dB` digital gain;
- verified capture/playback overlap and recovered alignment chirp; and
- zero clipped samples.

Failure of authorization, preflight, device provenance, alignment, playback,
capture, or scoring aborts the live result. Replay may still be run separately;
it may never substitute for a failed live condition.

## 5. Presentation contract

The dashboard must provide:

- teacher, rest, learned, and per-attempt audio controls;
- a 96-prototype bounded-memory map;
- the complete best-distance curve and visible exploration/recall phase change;
- a five-act event timeline driven by report data;
- exact-efference identifiers and hashes;
- the four scored workspace outcomes and their threshold-relative distances;
- source routing and geometry;
- the retained Stage 6 latency with its non-current-session label; and
- the claim boundary beside the final pass/fail state.

The dashboard must remain usable at 375 px, 768 px, and desktop widths and must
not require a network connection.

## 6. Success and claim boundary

`demo_passed` means that the selected retained association, exact-efference
panel, artifact integrity, bounded memory, and dashboard contract all passed in
the named execution mode.

A pass demonstrates an integrated bounded story: target-specific prototype
association, deterministic recall, exact commanded self playback, and
fixed-geometry self/world workspace handling. It does not demonstrate infant
learning, phoneme recognition, semantics, conversation, continuous inverse
closure, zero-shot or generalized self recognition, arbitrary rooms or
speakers, unknown-timing world dropout, or a new H2/H3/H4/H5 promotion.
