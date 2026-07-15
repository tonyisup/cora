# Cora-Infans

Cora-Infans is a bounded local auditory system whose primary product direction
is authorized, target-specific project-owner vocal imitation over a fixed finite
repertoire. Its supporting infrastructure includes environmental vocal-activity
gating, predictive event tracking, exact-efference workspace binding, and
deterministic synthetic-plant validation.

The governing operational contract is [VISION.md](VISION.md). It distinguishes
implemented mechanisms from measured capabilities and explicitly stopped
research families.

External corpus use follows [docs/DATA_POLICY.md](docs/DATA_POLICY.md): Cora is
a personal, non-commercial research project, and third-party audio stays outside
the repository.

## Current evidence status

| Stage | Status | Honest interpretation |
|---|---|---|
| Reproducible substrate | Implemented | Synthetic/live contracts, deterministic reports, and tests exist |
| Real-audio evaluation | First replay passed | WAV and prior-art captured NumPy PCM are measured directly with dev/held-out provenance; developmental-speech breadth is still open |
| Environmental vocal gating | Broad register detection falsified | Absolute, adaptive, temporal, predictive, cohort, and local nonlinear mechanisms failed reliable cross-cultural IDS/ADS detection. The five-second vocal front gate and relational cohort effect remain narrow filters; active register research is stopped and USD/TOR untouched |
| Local predictive hierarchy | H2 supported within tested boundaries | Natural event transfer replicated across Wellington and MEN; a separate real room run with known efference masks retained four states through 10/15/20% dropout while improving vowels `+0.614` and tones `+0.474`. Unknown-timing world dropout and semantics remain outside scope |
| Workspace binding | H3 supported within tested boundaries | Structured tones and exact-utterance natural speech passed at zero clipping. A later no-replay same-source forward model failed on three unseen commands at F1/recall/specificity `0.727/0.667/0.667`, so the exact-efference boundary remains |
| Synthetic plant infrastructure | Plant supported; local inverse falsified | The deterministic five-gesture plant passed, while hard and smooth local learners preserved `20/20` vowel identities but failed continuous closure. The plant remains infrastructure; categorical or global inverse claims require a VISION revision |
| Bounded vocal imitation | Primary product direction | The retained project-owner V4 result passed frozen development and held confirmation for `/a/`, `/i/`, and `/u/`. It supports finite target-specific association and recall only—not H4 closure, phoneme recognition, or speaker generalization |
| Efficiency | H5 supported within tested boundary | A six-fieldsite speech panel kept salient p95 at 10.17–10.38 ms with clean playback intervals, but two runs clipped only in trailing quiet and failed whole-capture integrity. Same-host energy was 27.36 µJ/frame versus 11.08 µJ/frame for the simpler baseline |

New to the terminology? Start with the [data dictionary](docs/DATA_DICTIONARY.md),
especially the distinction between the classifier **F1 score** and acoustic
**F1/F2/F3 formants**.
The current stage-by-stage claim boundary is summarized in the
[VISION evidence audit](docs/VISION_EVIDENCE_AUDIT_2026-07-11.md).

Inspect the machine-enforced contract:

```bash
python3 -m cora_infans --show-operational-contract
```

Verify that every bounded operational stage has its required retained evidence
and that every artifact matches its frozen SHA-256 pin:

```bash
python3 -m cora_infans --operational-readiness
```

This is a local-only project: run readiness and the hardware-safe suite before
changing its operational claims. Remote CI is intentionally not configured;
live hardware and protected corpora are never accessed by automated checks.

Check the current host and XVF3800 without opening an audio stream:

```bash
python3 -m cora_infans --operational-preflight \
  --manifest config/operational-point.json
```

The manifest records the latest session-specific 50%-knob confirmation. Reset
and reconfirm it before a future physical acceptance session.

The bounded live procedure, output files, stop rules, and recovery steps are in
the [operational runbook](docs/OPERATIONAL_RUNBOOK.md). Live acceptance requires
an explicit `--authorize-live-io` flag and refuses before playback when
preflight or stimulus provenance fails.

## Integrated progress hero demo

Generate the hardware-safe, deterministic version of the complete progress
story:

```bash
python3 -m cora_infans --hero-demo \
  --source synthetic \
  --mode replay \
  --vowel a \
  --output-dir /private/tmp/cora-hero-demo
```

When the retained project-owner V4 session is available, use its real target,
frozen learning curve, and learned association instead:

```bash
python3 -m cora_infans --hero-demo \
  --source vocal-session \
  --session-dir /private/tmp/cora-vocal-imitation-v4 \
  --mode replay \
  --vowel a \
  --output-dir /private/tmp/cora-hero-demo-vocal
```

Open `hero-demo.html`. The five-act dashboard exposes the teacher and learned
voices, all 12 motor attempts, the 96-prototype memory, final-three recall,
exact-efference identity, self/world overlap, wrong-efference control, artifact
hashes, and the retained Stage 6 latency. Replay mode uses a deterministic room
fixture and is presentation evidence only; it never claims live hardware.

The separately gated live adapter uses Mac mini Speakers as self device 3,
External Headphones as world device 2, and XVF3800 input `:0` at the standing
geometry. It requires both a passing read-only preflight and literal
`--authorize-live-io`; do not use that flag for ordinary dashboard generation.
The frozen [hero demo specification](docs/HERO_DEMO_SPEC.md) defines the control
panel, playback bound, stop rules, and claim boundary.

## Primary product: bounded vocal imitation

The primary product direction is an authorized, target-specific project-owner
vocal-imitation session over a fixed finite repertoire. The completed V4 result
is retained evidence, not an open-ended model or a reusable training set.

The product report will keep **imitation fidelity** (captured external target ↔
captured agent response) separate from **self-model fidelity** (predicted agent
capture ↔ actual agent response). It will visibly distinguish retained V4
evidence, synthetic fixtures, and future explicitly authorized real sessions.

Read [the bounded vocal-imitation product specification](docs/BOUNDED_VOCAL_IMITATION_PRODUCT_SPEC.md)
before creating a future session. It preserves the Stage 2/H4 stops, protected
corpus boundaries, raw-audio-outside-Git policy, and the requirement for a new
versioned protocol before any new owner recording.

## Bounded speech-learning demo

Run the offline cross-configuration imitation demo:

```bash
python3 -m cora_infans --demo-speech-learning \
  --scenario cross-configuration \
  --vowel all \
  --output-dir /private/tmp/cora-speech-learning-demo
```

The agent babbles a fixed 96-action motor repertoire, performs label-blind local
prototype search for `/a/`, `/i/`, and `/u/`, stores one bounded association per
target, and recalls it for the final three attempts. The output directory
contains target/learned WAV pairs and `speech-learning-session.json` with the
complete learning curve and hashes. Open `speech-learning-demo.html` for audio
controls and a compact per-target learning dashboard.

The governing [demo specification](docs/DEMO_SPEC.md) keeps this separate from
continuous inverse closure, human speech learning, phoneme recognition, and H4
evidence. It performs no live I/O and accesses no recorded or protected audio.

### Project-owner vocal targets

The completed bounded stage records two development takes and one untouched
confirmation take for each of `/a/`, `/i/`, and `/u/`. To create a fresh
successor session under its stop rules, say a comfortable steady vowel for
roughly two seconds for every capture:

```bash
python3 -m cora_infans --capture-vocal-imitation \
  --session-dir /private/tmp/cora-vocal-imitation \
  --target a --role development --take 1 \
  --authorize-project-owner-recording
```

Repeat development take `2`, then do both development takes for `i` and `u`.
Capture uses the XVF3800 callback at 16 kHz, retains the two
fused host channels, hashes every WAV, and refuses duplicates. Begin the vowel
after the short start tone from the Mac mini Speakers; the cue is discarded
before the retained microphone window.

Freeze all three learned associations before any confirmation recording:

```bash
python3 -m cora_infans --freeze-vocal-imitation-development \
  --session-dir /private/tmp/cora-vocal-imitation
```

The command refuses unless exactly six eligible development takes exist. Once
it passes, capture confirmation take `1` for each target. The capture command
itself refuses confirmation until the frozen-association artifact verifies.

After all nine takes exist, run the selection-blind confirmation:

```bash
python3 -m cora_infans --run-vocal-imitation \
  --session-dir /private/tmp/cora-vocal-imitation
```

This produces learned WAVs, `vocal-imitation-result.json`, and
`vocal-imitation-demo.html`. The frozen
[project-owner protocol](docs/VOCAL_IMITATION_PROTOCOL.md) requires every motor
association to freeze before confirmation audio is decoded. This is a
one-speaker demo boundary, not H4 or general speech-learning evidence.

The fresh V4 project-owner run passed all development and held-confirmation
gates. Its auditable metrics and artifact hashes are retained in
[`docs/evidence/2026-07-13-vocal-imitation-v4-real-result.json`](docs/evidence/2026-07-13-vocal-imitation-v4-real-result.json);
raw owner recordings and generated WAVs remain outside Git.

Validate a future flat Stage 6 run summary against the named 10.5 ms operating
point:

```bash
python3 -m cora_infans --validate-stage6-operating-point --report run.json
```

The report must include the exact salient ADC-frame-end metric and all zero-
count integrity fields listed by the contract. Historical experiment gates are
retained unchanged.

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
register scorer passed a preregistered 24-clip Mentawai/Toposa test with F1
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
not. The final governance decision records broad cross-cultural Stage 2
detection as falsified while retaining the front gate as narrow evidence.

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

A second, independently frozen development test used 500 aligned ordinary
read-speech vowel tokens from five CMU Arctic speakers. The same shared LPC
formant route peaked at 57.2% (120 ms, order 20); `/o/` and `/u/` recall were
only 13% and 30%. The five CMU Arctic confirmation speakers remain unfetched.
The subsequent continuous inverse-synthesis candidate also failed, despite
lowering reconstruction loss. A genuinely different three-gesture geometric
candidate then passed held-out local forward prediction but failed its frozen
synthetic plant and inverse-closure gates: `/u/` collapsed to `/e/`, identity
was `0.80`, agreement `0.60`, and F3 MAE `56 Hz`. The current mechanism fork,
retained outcomes, and protected-data boundary are summarized in
[docs/H4_MECHANISM_DECISION.md](docs/H4_MECHANISM_DECISION.md). No new H4
experiment or confirmation-data access is authorized.

Synthesize and inspect one vowel:

```bash
python3 -m cora_infans --synthesize-vowel a --out /tmp/a.wav --analyze
```

Benchmark only the in-process event pipeline:

```bash
python3 -m cora_infans --benchmark-core --iterations 1000 --frame-ms 25
```

Run one sustained processor workload for an external same-host power sampler:

```bash
python3 -m cora_infans --energy-workload --mode cora --duration-seconds 10
python3 -m cora_infans --energy-workload --mode conventional --duration-seconds 10
```

These commands report throughput and process resources but do not measure power
themselves. The frozen energy protocol uses privileged macOS `powermetrics` and
must not be interpreted as wall-plug or cross-device energy.

Audit a completed external nine-trial directory:

```bash
python3 scripts/audit_stage6_energy.py /private/tmp/cora-stage6-energy-TIMESTAMP
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
python3 -m cora_infans --listen-live --duration-ms 1000 --frame-ms 25 --sample-rate 16000 --capture-backend callback
python3 -m cora_infans --snapshot-firmware
python3 -m cora_infans --tune-raw-linear
```

Every listener report includes host-monotonic capture-call, pipeline-decision,
deadline-miss, real-time-factor, process CPU, process-lifetime peak RSS, and
native input-overflow metrics. Blocking backends measure the start of
`capture_frame()` to the workspace decision. The callback backend additionally
preserves PortAudio's ADC-buffer timestamp and reports frame onset/end to the
workspace decision.

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

The callback backend preserves PortAudio's ADC buffer timestamp and reblocks
variable callback quanta into exact analysis frames. In three frozen quiet-path
runs, ADC frame-end-to-workspace p95 was 10.19–10.21 ms with zero 30 ms device
deadline misses. Frame-onset latency was 35.19–35.21 ms because it correctly
includes the 25 ms needed to acquire a complete frame. These are backend ADC
timestamps, not raw CoreAudio timestamps, and they exclude XMOS internal DSP.

The structured-salient retry uses the external-world speaker at 180 degrees and
50 cm with a fixed pitch contour at -12 dB. Across three final runs it emitted
142 salient events, clipped zero of 960,000 captured samples, and kept salient
ADC frame-end-to-workspace p95 between 9.94 and 10.20 ms with zero device
deadline misses. This is structured-tone timing, not natural-speech timing.

The same-host sustained energy comparison found similar estimated CPU power for
Cora and the simpler conventional baseline (13.66 W versus 13.53 W), but the
baseline processed 2.44 times as many frames per second. Cora therefore used
27.36 J per million frames versus 11.08 J, or 2.47 times the energy per frame.
Raw power files remain outside Git with a retained hash manifest. This is a
`powermetrics` optimization estimate—not wall-plug, acoustic, XVF3800, paced
live-runtime, or cross-device energy.

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
- A six-phase, order-preserving trajectory model was then evaluated with
  leave-one-fieldsite-out fitting on all 200 already-spent vocal clips. It
  reached F1 `0.302` at zero thresholds and only `0.289` after out-of-field
  threshold selection under the existing FPR ceilings. This workflow accessed
  no Wellington, San Diego, or Toronto audio; Wellington was nevertheless
  already spent for a separate H2 task. Another attempt must change the
  temporal learning mechanism or protocol rather than retune this candidate.
- An authorized predictive redesign retained every ordered 50 ms prosodic
  state and scored sequences with four class-conditional linear-Gaussian
  dynamics. On the same 200 spent clips it improved F1 from `0.289` to `0.545`
  while keeping FPR `0.099` and song FPR `0.071`, but recall was only `0.466`
  and ACO/ARA remained at F1 `0.0`. The frozen stop rule blocks parameter
  tuning and further WEL/USD/TOR access. San Diego and Toronto remain pristine;
  Wellington is already fieldsite-spent by H2 and cannot serve as pristine
  Stage 2 confirmation.
- A label-free cohort-adaptive revision aligned held-out transition geometry
  using at least three target speakers and known vocal-mode grouping. On 198
  eligible clips it improved selected F1 to `0.681`, precision to `0.865`, and
  FPR to `0.035`, but recall remained `0.561`; ACO stayed at F1 `0.0` and four
  other fieldsites missed the frozen floor. The gain is meaningful but does not
  rescue cross-cultural promotion, and USD/TOR remain untouched.
- An authorized nonlinear candidate learned 32 temporal prototypes from
  label-blind 200 ms contexts using winner-local updates, then used Gaussian
  radial-basis responses in condition-specific predictive readouts. All folds
  used every prototype and reduced quantization error, but selected F1 was only
  `0.535`, recall `0.466`, FPR `0.113`, and song FPR `0.083`; ACO and ARA stayed
  at F1 `0.0`. It slightly underperformed the same-data linear model, so its
  frozen stop rule blocks codebook/readout tuning and USD/TOR access.
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
  fixed-room, fixed-geometry boundary.
- The natural-speech extension retained two preregistered failures before its
  final result. Chirp-conditioned FIR subtraction had no development gap and
  clipped three captures. A raw prior-replay PCM template narrowed the gap to
  `-0.0093` but still failed and clipped one overlap. The final candidate froze
  a phase- and gain-invariant 12-band spectral-shape distance and its numeric
  threshold on those now-spent captures, then evaluated three fresh source
  pairs once. Fresh repeated-self distances were `0.0687/0.0100/0.0419`, all
  below the `0.0790` threshold; overlaps were `0.4409/0.2357/0.1481`, all above
  it. Scheduled capacity-3 F1/recall/specificity were `1.0/1.0/1.0`, flat F1
  was `0.8`, fixed-window world recall was `0.5`, capacity `1/2/3/4` reached
  `0.857/0.923/1.0/1.0`, and all 12 captures had zero clipping. This extends
  H3 only to exact-efference speech binding after one prior replay at the same
  fixed room and geometry; generalized self voice, arbitrary events, geometry
  transfer, semantics, and neural dynamics remain untested.
- The next H3 candidate learned one 12-band self-source path signature plus a
  count scalar from nine prior commands and stored no target-utterance replay.
  Its spent leave-one-command-out gate passed, but the preregistered fresh live
  run failed: two overlaps scored below the `0.3006` threshold and one self-only
  probe scored above it. Capacity-3 scheduled F1/recall/specificity were
  `0.727/0.667/0.667`, below flat F1 `0.8`; all nine captures were aligned and
  zero-clipping. Pairwise overlap distance still exceeded self distance for all
  three commands, but between-command offsets eliminated a global threshold.
  This descriptive structure cannot rescue the frozen no-calibration protocol.
- Theta/gamma scheduling is an event-timing model, not measured neural dynamics.
- The vowel table uses adult reference targets and clean synthetic success does
  not imply recorded infant/adult speech performance.
- The approved continuous inverse-synthesis H4 candidate is retained as a
  preregistered failure. Its local updates strictly reduced spectral
  reconstruction loss on every recorded development example, but identity
  reached only `0.85` synthetic, `0.44` VocalSet, and `0.574` CMU Arctic versus
  `0.90/0.80/0.80` gates. Lower reconstruction error therefore did not establish
  a useful shared perception-production representation; protected confirmation
  speakers remain untouched.
- Process CPU/RSS, native-overflow proxy, host/device-boundary latency, and an
  offline conventional comparison are measured. The quiet-path device timing
  and structured-salient timing gates passed; same-host sustained estimated
  energy is measured. A natural-speech timing attempt passed latency but failed
  capture quality at maximum speaker volume; a separately frozen diagnostic at
  approximately 50% hardware volume then passed every gate across 1,200 frames
  with per-frame clipping reconciliation. A later six-fieldsite panel passed
  every timing and playback-interval clipping gate at salient p95
  `10.17–10.38 ms`, but KRA and MES had full-scale events only during trailing
  quiet, failing the frozen whole-capture rule. Raw CoreAudio timestamps, XMOS
  internal DSP latency, and embodied/paced-live energy remain open.

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
- `natural_speech_workspace_spectral_v3.py` — final bounded 12-band exact-efference speech-binding gate
- `same_source_forward_model.py` — bounded command-conditioned self-path model
- `same_source_forward_live.py` — fresh no-replay same-source H3 gate
- `register_saliency.py` — long-window vocal, speech/song, and adaptive IDS gates
- `io/motor_plan.py` — source-filter synthesis, LPC perception, and vowel gate
- `continuous_inverse_synthesis.py` — frozen bounded H4 continuous motor search
- `geometric_articulatory_synthesis.py` — frozen H4 geometric plant, local forward learner, and inverse gate
- `benchmark.py` — explicit-boundary event-pipeline microbenchmark
- `pipeline.py` — saliency, predictive state, and workspace admission
- `theta_gamma.py` — deterministic bounded temporal scheduler
- `io/live_mic.py` — real XVF3800/CoreAudio adapter
- `io/active_attention.py` — pure beam-steering governor
- `io/hardware_tuner.py` — verified USB DSP controls

Historical experiment logs and retractions are retained under `docs/`. The
current claim boundary is defined by `VISION.md` and this README.

The frozen Option D mechanism and its retained synthetic failure are documented
in [Option D: geometrically constrained articulatory synthesis](docs/H4_OPTION_D_GEOMETRIC_ARTICULATORY_SYNTHESIS.md).
It does not authorize a retry or recorded-speech access.

The subsequent [Option E factorized plant](docs/H4_OPTION_E_FACTORIZED_FIVE_GESTURE_PLANT.md)
passed its intended five-vowel expressivity test but failed a stricter all-state
resonance gate. Its stop rule fired before forward learning or inverse search.

A disjoint [Option F expanded-band confirmation](docs/H4_OPTION_F_EXPANDED_BAND_PLANT.md)
then passed every plant gate. It authorizes only a new synthetic learner
preregistration; H4 and recorded-speech access remain unpromoted.

The resulting [Option G local-closure candidate](docs/H4_OPTION_G_EXPANDED_LOCAL_CLOSURE.md)
retained perfect post-hoc identity but failed continuous forward accuracy and
equivalence-aware reconstruction. Its spent points cannot be used for tuning.

The disjoint smooth-local Option H candidate also failed and often underperformed
its nearest-exemplar baseline. Further local kernel/density/search retries are
stopped. A global nonlinear representation or relaxed closure claim would
require a revision to the governing operational specification.

The final [H4 governance decision](docs/evidence/2026-07-11-h4-bounded-local-family-disposition.json)
records the bounded-local representation family as falsified within tested
scope. Active H4 work is stopped and all reserved confirmation data remains
untouched.
