# Cora-Infans

Cora-Infans is an audio-centric research system for testing local auditory
learning, bounded event memory, active attention, and analysis by synthesis.

The governing research contract is [VISION.md](VISION.md). It distinguishes
implemented mechanisms from measured and validated capabilities.

External corpus use follows [docs/DATA_POLICY.md](docs/DATA_POLICY.md): Cora is
a personal, non-commercial research project, and third-party audio stays outside
the repository.

## Current evidence status

| Stage | Status | Honest interpretation |
|---|---|---|
| Reproducible substrate | Implemented | Synthetic/live contracts, deterministic reports, and tests exist |
| Real-audio evaluation | First replay passed | WAV and prior-art captured NumPy PCM are measured directly with dev/held-out provenance; developmental-speech breadth is still open |
| Parentese-relevant saliency | Mixed; not promoted | The final 16-fieldsite refit reached F1 0.600 on untouched Enga/Turku clips and failed its 0.65 gate; the separately tested vocal front gate remains positive |
| Local predictive hierarchy | H2 supported within tested boundaries | Natural event transfer replicated across Wellington and MEN; a separate real room run with known efference masks retained four states through 10/15/20% dropout while improving vowels `+0.614` and tones `+0.474`. Unknown-timing world dropout and semantics remain outside scope |
| Workspace binding | H3 supported within tested boundaries | The preregistered -18 dB synchronized run passed every classifier and zero-clipping gate: scheduled F1/recall/specificity 1.0/1.0/1.0 versus flat F1 0.8 and fixed-window F1 0.667; capacity improved through three slots and plateaued at four |
| Analysis by synthesis | Recorded-human gate failed; retry paused | Clean synthesis remained 100%, but the preregistered VocalSet confirmation scored 36%; a disjoint development ladder peaked at 56%, so the reserved confirmation singers remain untouched and H4 stays open |
| Efficiency | Resources measured; deadline gate failed | Three native XVF3800 runs had zero overflow flags, RTF below 1.0, CPU below 3.8%, and RSS below 63 MiB, but 11.75–18.5% exceeded the 30 ms operational deadline; ADC-onset timestamps and energy remain open |

## Hardware boundary

The XVF3800 board has four physical microphones and performs on-board DSP. The
current macOS/CoreAudio path exposes two fused USB channels. Cora therefore does
not claim raw four-channel host-side learning or beamforming.

Firmware speech-classified source direction can drive beam steering when it is
finite. Two-channel energy imbalance is otherwise only a coarse left/right cue.

## Install and test

```bash
python3 -m pip install -e .
python3 -m pytest tests -q
```

The installed entry point is `cora-infans`; `python3 -m cora_infans` is the
portable fallback.

## Research commands

Describe implemented and planned surfaces:

```bash
python3 -m cora_infans --describe
```

Run deterministic pipeline scenarios:

```bash
python3 -m cora_infans --simulate-frame
python3 -m cora_infans --simulate-sequence --family gap
```

Evaluate real uncompressed WAV clips (integer PCM or IEEE float):

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --frame-ms 250 \
  --saliency-threshold 0.02 \
  --score-mode peak_rms
```

The label CSV maps annotations to existing WAV files. `split` is required for
development/held-out comparison:

```csv
clip_id,start_s,end_s,label,split
parentese_dev_01,0.20,1.10,parentese,dev
parentese_held_01,0.15,0.95,parentese,held_out
```

Annotations are used only as ground truth. WAV samples—not labels—produce the
evaluated observations.

Tune transparent saliency baselines on `dev` and evaluate once on `held_out`:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --compare-baselines \
  --positive-label infant_directed_speech
```

This compares peak RMS, spatial contrast, pitch, pitch confidence, spectral
centroid, and spectral flux. Every threshold is selected on development clips
and frozen for held-out evaluation.

### External developmental-speech corpus

The first corpus is the Music Lab's [Human vocalization corpus](https://zenodo.org/records/5525161)
(DOI `10.5281/zenodo.5525161`). It contains infant-directed and adult-directed
speech/song from 21 societies and is licensed CC BY-NC-SA 4.0. The edited audio
archive is approximately 5.8 GB, so Cora does not download it automatically.

After intentionally downloading and extracting the edited archive, build a
participant-grouped manifest:

```bash
python3 -m cora_infans --prepare-ids-manifest \
  --root /path/to/IDS-corpus-edited \
  --out /path/to/IDS-corpus-edited/labels.csv
```

The importer maps filename condition `B` to `infant_directed_speech`, condition
`D` to `adult_directed_speech`, and assigns all clips from one participant to
the same development or held-out split. Song conditions are excluded unless
`--include-song` is requested.

The first 12-clip participant-disjoint pilot is recorded in
`docs/evidence/2026-07-10-ids-pilot.json`. It is deliberately not a promotion:
the best feature's greater recall came with an unacceptable false-positive
increase. Pilot participants are excluded from confirmatory reuse.

The criterion was then frozen and evaluated on 12 fresh Beijing clips. Pitch
confidence passed narrowly: held-out F1 `0.575` versus RMS `0.510`, with FPR
`0.594` versus `0.547`. The auditable result is in
`docs/evidence/2026-07-10-ids-confirmation.json`. It supports a one-fieldsite,
participant-disjoint result—not a robust universal parentese detector.

That limitation is now measured rather than hypothetical. A preregistered test
froze the Beijing thresholds and added paired song controls for unseen Krakow
and Mbendjele BaYaka speakers. Both fieldsites failed. See
`docs/evidence/2026-07-10-cross-fieldsite-result.json`.

The replacement is a transparent long-window pipeline rather than another
absolute pitch threshold. It first uses YIN voiced-window coverage to reject
non-vocal clips, then separates speech from song, then compares IDS with ADS
after unlabeled normalization within speaker and vocal mode. The corrected
register scorer passed a preregistered 24-clip Menominee/Toposa test with F1
`0.667`, FPR `0.111`, and zero song false positives. Its calibration requires
two observations from the same speaker and mode, so it is not cold-start or
strictly causal.

Environmental controls were tested separately. The first vocal gate failed;
the frozen revision retained 12/12 fresh Hadza vocal clips and rejected 15/16
fresh ESC-50 fold-3 controls. The remaining false positive was crackling fire.
The composed, hash-pinned artifact is
`docs/evidence/2026-07-10-register-pipeline-v4.json`. A later jointly untouched
Vanuatu/ESC-50 fold-4 test retained 12/12 vocals and rejected 16/16
environmental clips, but the register scorer fell to F1 `0.400` and failed its
frozen `0.65` criterion. The vocal front gate generalized; IDS/ADS scoring did
not. Stage 2 therefore remains unpromoted.

A grouped-training threshold revision then reached F1 `0.857` on fresh
Nyangatom clips but failed the strict song-control ceiling by promoting one of
six songs. The final same-family attempt refit unchanged transparent features
on 176 spent clips from 16 fieldsites. On untouched Enga and Turku clips it
reached pooled F1 `0.600`, below the frozen `0.65` gate, despite FPR `0.056`,
song FPR `0.083`, and vocal recall `1.0`. Its preregistered stop rule is active:
remaining corpus fieldsites will not fund another threshold/refit iteration of
this model family.

Fit a self-contained scorer from external, spent corpus subsets and evaluate a
frozen artifact on an unseen speaker batch:

```bash
python3 -m cora_infans --fit-register-scorer \
  --source /path/to/subset::/path/to/subset/labels.csv \
  --out /tmp/register-model.json

python3 -m cora_infans --eval-register-scorer \
  --model /tmp/register-model.json \
  --root /path/to/unseen-subset \
  --label-csv /path/to/unseen-subset/labels.csv
```

Evaluate prequential prediction on held-out acoustic observations:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
    --evaluate-prediction dominant_frequency_hz
```

Evaluate the bounded online learner's safety separately from superiority:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --evaluate-online-safety peak_rms,pitch_confidence,spectral_centroid_hz,spectral_flux
```

The safety report compares every clip with persistence and EMA, reporting mean
and worst RMSE regret separately from strict win rate. Passing safety does not
count as predictive advantage.

Plan the next structured continuous-audio experiment without touching audio
devices:

```bash
python3 -m cora_infans --evaluate-structured-prediction \
  --out-dir /tmp/cora-structured
```

Exercise its complete synthetic fake-I/O path:

```bash
python3 -m cora_infans --evaluate-structured-prediction \
  --out-dir /tmp/cora-structured-fake \
  --fake-io
```

Fake success cannot promote predictive evidence. Real execution requires an
explicit authorization flag plus recorded input/output devices, speaker
distance/orientation, and quiet-room confirmation. The frozen protocol and
acceptance artifacts are under `docs/evidence/2026-07-10-structured-*`. The
first real-I/O attempt stopped before playback on a diagnosed ffmpeg buffering
issue. The separately frozen and authorized retry completed: tones passed, but
vowels failed, so the overall two-family acceptance rule did not pass.
A development-only calibration redesign then advanced to entirely fresh seeds
and passed the controlled real two-family predictive-advantage gate. This does
not yet establish unconstrained or cross-speaker speech prediction.

Audit a local directory of paired natural speech clips without using condition
labels in learning or scoring:

```bash
python3 -m cora_infans --evaluate-natural-prediction-design \
  --root /path/to/spent-development-wavs \
  --participants BEJ06,BEJ07,BEJ11,BEJ21
```

The default grid is a design audit only and cannot promote spent audio.

Evaluate the frozen causal event representation across fieldsites:

```bash
python3 -m cora_infans --evaluate-natural-event-transfer \
  --development-root /path/to/beijing-development-wavs \
  --development-participants BEJ06,BEJ07,BEJ11,BEJ21 \
  --held-out-root /path/to/wellington-held-out-wavs \
  --held-out-participants WEL01,WEL02,WEL03,WEL04
```

The retained preregistration and result record all external WAV hashes. The
learner and scoring never consume the B/D condition labels.

Post-confirmation component ablations are also available:

```bash
python3 -m cora_infans --evaluate-natural-event-ablations \
  --development-root /path/to/beijing-development-wavs \
  --development-participants BEJ06,BEJ07,BEJ11,BEJ21 \
  --held-out-root /path/to/spent-wellington-wavs \
  --held-out-participants WEL01,WEL02,WEL03,WEL04
```

These explain the retained result; they cannot promote it again.

Replay retained Corollary real-capture arrays without copying them:

```bash
python3 -m cora_infans --prepare-corollary-manifest \
  --root /path/to/corollary/capture-family \
  --out /tmp/cora-corollary.csv

python3 -m cora_infans --eval-corpus \
  --root /path/to/corollary/capture-family \
  --label-csv /tmp/cora-corollary.csv \
  --input-mode recorded_npy \
  --compare-baselines \
  --positive-label mixed_voice
```

The importer uses seeds 0/1 for development and seed 2 for held-out evaluation
by default. This evidence is real captured audio but belongs to a repeated,
aligned self/world experiment, not an infant-directed-speech corpus.

Measure the clean synthetic vowel confusion matrix:

```bash
python3 -m cora_infans --evaluate-vowel-loop
```

Evaluate the preregistered five-vowel VocalSet confirmation subset after its
external manifest has been fetched:

```bash
python3 -m cora_infans --evaluate-recorded-vowels \
  --root /external/vocalset-subset \
  --manifest /external/vocalset-manifest.json
```

VocalSet is CC BY 4.0 and remains outside Git. The frozen subset contains five
untouched singers × `a/e/i/o/u`; filenames and hashes are retained, while
vowel labels enter scoring only after label-blind LPC analysis.

That confirmation failed at 36%. On five disjoint development singers,
speaker normalization, LPC-order selection, continuous spectral-to-formant
mapping, and spectral centroids reached at most 28%, 40%, 52%, and 56%
respectively under leave-one-singer-out scoring. Because none approached the
frozen 80% gate, the next reserved five-singer confirmation set has not been
fetched or inspected.

Synthesize and inspect one vowel:

```bash
python3 -m cora_infans --synthesize-vowel a --out /tmp/a.wav --analyze
```

Benchmark only the in-process event pipeline:

```bash
python3 -m cora_infans --benchmark-core --iterations 1000 --frame-ms 25
```

The benchmark report lists all excluded latency sources and does not measure
energy.

Run the workspace strategy ablation:

```bash
python3 -m cora_infans --evaluate-workspace-ablation
```

Run the non-promoting real-capture H3 design audit against external retained
Corollary artifacts:

```bash
python3 -m cora_infans --evaluate-real-workspace-design \
  --root /path/to/corollary/archive/data/m2_baseline
```

This consumes Corollary-produced self-presence and speech summary features plus
verifies the underlying real PCM hashes. It does not recompute a signal residual
or replace a fresh Cora synchronized binding task.

Plan or exercise the device-agnostic structured H3 protocol:

```bash
python3 -m cora_infans --evaluate-structured-workspace-binding
python3 -m cora_infans --evaluate-structured-workspace-binding --fake-io
```

The protocol uses held-out self/world frequency bands, self-only/world-only/
overlap conditions, flat and fixed-window baselines, and a 1–4 binding-slot
sweep. Generic dual-track concurrency and chirp-aligned two-channel capture
evaluation are tested. A real-I/O CLI path is now wired behind a fail-closed,
hash-bound preregistration gate. The final preregistration freezes the self
source centered directly below the array at 4 cm, -90 degrees elevation, and
100% system volume. It also freezes device 3 for self, device 2 for world, and
XVF3800 input `:0`. It was committed before the first live H3 execution.

The ablation compares flat, fixed-window, and theta-gamma handling on aligned
self echo and legitimate world/motor overlap. Scheduling suppression requires
verified self-audio identity; temporal coincidence by itself is insufficient.

## Live hardware commands

```bash
python3 -m cora_infans --capture-live --device :0 --duration-ms 250
python3 -m cora_infans --listen-live --duration-ms 2000 --with-attention
python3 -m cora_infans --listen-live --duration-ms 1000 --frame-ms 25 --sample-rate 16000 --capture-backend native
python3 -m cora_infans --snapshot-firmware
python3 -m cora_infans --tune-raw-linear
```

Every listener report includes host-monotonic capture-call, pipeline-decision,
deadline-miss, real-time-factor, process CPU, process-lifetime peak RSS, and
native input-overflow metrics. The boundary is the start of `capture_frame()`
to the workspace decision—not ADC sample onset or an audio-backend timestamp.

When the native backend provides a nominal input-latency value, the report also
adds it to the host call measurements as an explicitly estimated
ADC-to-workspace distribution. It is not presented as a per-buffer hardware
timestamp.

The CLI listener prefers a native PortAudio/CoreAudio stream through
`sounddevice`, using the same substrate validated by the Corollary prior-art
project. `--capture-backend ffmpeg` is an explicit real-audio fallback. Auto
fallback reasons are included in reports; neither path substitutes synthetic
audio. Reports separate cold-start latency from steady state and expose both
strict and jitter-tolerant deadline misses.

The frozen Stage 6 resource run completed three native 10-second replicates
without overflow or fallback. CPU, RSS, and real-time-factor criteria passed,
but the 30 ms operational deadline-miss rate was 11.75–18.5% versus a 5% gate.
All frames were quiet-path frames, so this does not establish salient-workload
timing. The offline Cora core was about 4.7 times slower at p95 than a much
simpler fixed-threshold/prototype baseline while remaining far below real time.

A single-variable native-rate retry reduced reported input latency from 95.3 ms
at 48 kHz to 42.0 ms at the XVF3800's native 16 kHz, lowering estimated
ADC-to-workspace p95 to roughly 73 ms. It did not improve the deadline gate:
14.25–18.5% of frames still exceeded 30 ms. The unresolved jitter is therefore
at the callback-quantum/blocking-read boundary rather than in pipeline compute
or host sample-rate conversion.

The bounded speaking and self-subtraction paths require explicit real-I/O
authorization. They must not be treated as evidence of conversational or
developmental language learning.

## Important limitations

- Licensed developmental-speech audio has been evaluated from external
  temporary subsets; the CC BY-NC-SA audio is not checked in.
- The original pitch-confidence threshold confused song with IDS and failed on
  both unseen fieldsites. Its speaker-and-mode adaptive replacement passed on
  two later fieldsites, then failed on Vanuatu; larger-data revisions failed a
  song ceiling and a final pooled-F1 gate. It requires batch calibration and is
  not a universal or cold-start parentese detector.
- The upstream vocal gate passed selected diffuse and impulsive controls but
  admitted one crackling-fire clip in fold 3; it rejected all selected fold-4
  controls in the later joint test.
- The current saliency gate is a loudness baseline, not a parentese detector.
- The online local predictor is now conservatively bounded and passed its fresh
  regret gate, but it still has zero real-clip wins. Only the onset-phase
  template has a held-out advantage, and that advantage excludes a later
  duration-normalized flux result retracted as a warm-up/clip-boundary artifact.
- The structured continuous-audio real run verified 12 synchronized captures.
  Tones passed every frozen gate (`0.860` accuracy vs `0.263` baseline), while
  vowels occupied only three states and tied their best baseline (`0.596`), so
  the first required two-family gate failed.
- A development-only unlabeled calibration normalization then passed on a new
  real 12-trial manifest: tones scored `0.895` versus `0.281`, vowels `0.702`
  versus `0.281`, and every held-out trial improved by at least `0.368`. This
  validates the bounded controlled mechanism; natural H2 remains open.
- A later fresh robustness run retained strong predictive advantage under an
  unseen `200 Hz` synthetic F0, `10 dB` noise, and `10%` dropout. F0/noise kept
  four vowel states, but dropout merged every `u` into the `i`-dominated state;
  the strict robustness verdict therefore fails despite `+0.579` improvement.
- A temporal-median dropout revision passed development folds but failed a
  fresh synthetic gate (`+0.193` pooled; `+0.105` at 15–20% dropout), so its
  real-I/O path is blocked and no additional live evidence was consumed.
- A later genuinely different candidate uses the motor/efference path's known
  erasure mask and averages only fully observed 30 ms frames. On the spent v4
  synthetic matrix it retained four states through 10/15/20% dropout and
  improved vowels by `+0.579/+0.526/+0.526`. This is design evidence only; a
  fresh synthetic gate and separate real preregistration are still required.
- That prospective gate then passed on six unused seeds: tones improved
  `+0.614` pooled and vowels `+0.596`, every 10/15/20% condition retained four
  states, and the minimum trial improvement was `+0.579`. This remains
  synthetic evidence; live execution requires a separate frozen protocol.
- The separately preregistered real run then completed all 12 synchronized
  playback/captures. Vowels retained four states and improved `+0.614` pooled
  (`+0.579` minimum trial); tones retained four states and improved `+0.474`
  (`+0.368` minimum). All 24 WAV hashes, alignments, stream positions, bounded
  state, and top-down use verified. This closes the tested known-efference
  dropout gate, not arbitrary-world or unknown-timing dropout.
- A label-blind natural-stream audit over eight spent Beijing speech clips then
  tested 18 frame/horizon variants. None advanced; the best won `2/8` clips,
  mean improvement was only `+0.0036`, and worst-clip regret was `9.9%`.
- A genuinely different causal event target was then selected on those spent
  clips and frozen before any Wellington audio was fetched. Trained on Beijing,
  its 16 transition counts beat the strongest causal baseline on all eight
  participant-disjoint Wellington clips: mean improvement `+0.0751`, worst
  improvement `+0.0107`, and at least three occupied states per clip. This is
  positive cross-fieldsite event-dynamics evidence, not content, semantic, or
  IDS/ADS prediction.
- Post-confirmation ablations on the now-spent Wellington clips support the
  mechanism's parts: the full model beat removal of bottom-up event context and
  removal of learned top-down transition use on `8/8` clips each. Hiding the
  quiet/saliency event reduced mean accuracy from `0.7600` to `0.6688`. These
  are descriptive component results, not another held-out promotion or an H3
  workspace test.
- A preregistered Beijing→ARA replication then failed before scoring: `ARA01B`
  is exactly as long as the mandatory 2-second calibration and `ARA03B` is only
  `1.632` seconds. The eight clips and hashes are retained, no participant was
  substituted, and calibration was not shortened. This is protocol-
  eligibility failure—not negative prediction evidence and not replication.
- A replacement protocol froze a `2.10 s` minimum-duration rule before access,
  selected MEN by repository/metadata order, and fetched candidates numerically
  until four paired participants qualified. MEN01–04 all qualified. With the
  unchanged Beijing-only model, all `8/8` MEN clips beat their strongest causal
  baseline; mean improvement was `+0.1106`, worst improvement `+0.0654`, and
  minimum occupancy `3`. This independently replicates natural speaker/
  fieldsite transfer for the event target.
- A spent real-capture H3 design audit selected thresholds on two Corollary
  families and held out a third. At three binding slots, flat admission scored
  F1 `0.727`; identity-only suppression preserved self specificity but reduced
  legitimate-world recall to `0.25`; combining bound self identity with bottom-
  up world-speech evidence scored F1 `1.0`. Capacity 1/2/3 improved
  `0.800/0.889/1.000` and plateaued at 4. This is design evidence over upstream
  summary features, not promoted H3 or fresh Cora audio.
- A new 15-trial structured H3 protocol then passed its synthetic acoustic
  plumbing gate: scheduled identity plus world evidence scored F1 `1.0` versus
  flat `0.8` and fixed-window `0.667`; capacity `1/2/3/4` scored
  `0.857/0.923/1.0/1.0`. The first symmetric midpoint threshold is retained as
  a failure because it suppressed two equal-gain overlaps; the corrected rule
  uses the lowest decile of the clean development gap. Live I/O remains unwired.
- Generic live plumbing now keeps self/world PCM separate, starts overlap tracks
  through a concurrent source abstraction, assigns one chirp reference source,
  and evaluates explicitly aligned mono or two-channel captures. Injected-I/O
  tests pass. A bounded acquisition runner now exercises all 15 trials through
  injected source-specific callbacks, retains separate self/world/capture WAVs
  and hashes, verifies capture stream position and chirp alignment, and aborts
  on source-start failure. It caps playback at 25 seconds and capture at 45
  seconds. The first preregistered real run completed all 15 trials and passed
  every classification gate: scheduled F1/recall/specificity were `1.0/1.0/1.0`
  versus flat F1 `0.8` and fixed-window recall `0.5`; capacity `1/2/3/4` scored
  `0.857/0.923/1.0/1.0`. A post-run quality audit found clipped samples in all
  nine self-bearing captures and none of six world-only captures, so its raw
  promotion flag is superseded and H3 is not yet promoted. Attempt 2 freezes
  self playback at `-12 dB` and adds a zero-clipping acceptance gate while
  preserving geometry, stimuli, threshold selection, and classification gates.
  It retained identical classifier metrics and reduced clipped samples from
  `2,287` to `124`, but still failed the zero-clipping gate. Attempt 3 freezes
  self playback at `-18 dB`; every other condition and criterion is unchanged.
  Attempt 3 again completed all 15 trials with identical classifier metrics and
  zero clipped samples across every capture. All original classification gates
  and the added quality gate passed, supporting H3 within this structured tone,
  fixed-room, fixed-geometry boundary. Natural speech, arbitrary spectra, and
  geometry transfer remain untested.
- Theta/gamma scheduling is an event-timing model, not measured neural dynamics.
- The vowel table uses adult reference targets and clean synthetic success does
  not imply recorded infant/adult speech performance.
- Process CPU/RSS, native-overflow proxy, host-boundary latency, and an offline
  conventional comparison are measured. The deadline gate failed; ADC-onset
  timing, salient live workload timing, and energy remain open.

## Main modules

- `io/wav_frames.py` — real uncompressed WAV decoding and measured features
- `io/recorded_frames.py` — retained real-capture arrays and prior-art manifests
- `io/offline_eval.py` — confusion metrics and dev/held-out baseline comparison
- `predictive_learning.py` — bounded local hierarchy and baseline evaluation
- `structured_prediction.py` — gated continuous-stream protocol and local state transitions
- `natural_prediction_audit.py` — participant-held-out, label-blind natural-stream design audit
- `natural_event_prediction.py` — causal quiet/onset/stable/change states and cross-fieldsite transfer
- `real_workspace_binding.py` — bounded self/world binding design audit over retained real captures
- `structured_workspace_binding.py` — device-agnostic self/world/overlap H3 protocol and fake gate
- `register_saliency.py` — long-window vocal, speech/song, and adaptive IDS gates
- `io/motor_plan.py` — source-filter synthesis, LPC perception, and vowel gate
- `benchmark.py` — explicit-boundary event-pipeline microbenchmark
- `pipeline.py` — saliency, predictive state, and workspace admission
- `theta_gamma.py` — deterministic bounded temporal scheduler
- `io/live_mic.py` — real XVF3800/CoreAudio adapter
- `io/active_attention.py` — pure beam-steering governor
- `io/hardware_tuner.py` — verified USB DSP controls

Historical experiment logs and retractions are retained under `docs/`. The
current claim boundary is defined by `VISION.md` and this README.
