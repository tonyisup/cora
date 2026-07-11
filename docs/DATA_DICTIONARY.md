# Cora-Infans data dictionary

This is the plain-language reference for acronyms, dataset codes, metrics, report
fields, hardware terms, and research-status words used throughout Cora-Infans.
It describes what terms mean **in this project**. A term may have a broader or
different meaning elsewhere.

## Start here: the most easily confused terms

| Term | Meaning in Cora-Infans |
|---|---|
| **ADS** | Adult-directed speech: speech addressed to another adult. |
| **IDS** | Infant-directed speech: speech addressed to an infant. “Parentese” is the informal concept; `IDS` is the dataset label. |
| **F1 score** | A classifier metric combining precision and recall. It ranges from `0` to `1`; higher is better. |
| **F1, F2, F3 formants** | The first three vocal-tract resonances, measured in hertz. These are acoustic features and are unrelated to the classifier’s F1 score. |
| **F0** | Fundamental frequency, usually perceived as pitch. |
| **FPR** | False-positive rate: the fraction of actual negatives incorrectly called positive. Lower is better. |
| **H1–H5** | The five research hypotheses in `VISION.md`. They are not the same numbering system as Stages 0–6. |
| **LOFO** | Leave-one-fieldsite-out evaluation: train without one fieldsite, test on that fieldsite, and repeat for every fieldsite. |
| **RMS** | Root mean square amplitude, used here as a simple signal-energy or loudness-related measure. |
| **p95** | The 95th percentile: 95% of measurements are at or below this value. |

## Dataset labels and identifiers

### Vocalization labels

| Label | Short form | Meaning | Positive for the Stage 2 IDS classifier? |
|---|---|---|---|
| `infant_directed_song` | ID song | Song directed to an infant | No; it is a song control. |
| `infant_directed_speech` | IDS | Speech directed to an infant | **Yes** |
| `adult_directed_song` | AD song | Song directed to an adult | No |
| `adult_directed_speech` | ADS | Speech directed to an adult | No |

For Stage 2 register reports, only `infant_directed_speech` is the positive
class. ADS and both song labels count as negatives.

### Music Lab filename format

The corpus uses `XXXYYZ.wav`:

- `XXX`: three-letter fieldsite code;
- `YY`: participant number, unique within that fieldsite;
- `Z`: vocalization type: `A` = infant-directed song, `B` = IDS,
  `C` = adult-directed song, `D` = ADS.

Example: `KRA01D.wav` is participant `01` from Krakow producing
adult-directed speech.

### Music Lab fieldsite codes

These names follow the [official corpus record](https://doi.org/10.5281/zenodo.5525161).

| Code | Fieldsite or society |
|---|---|
| `ACO` | Afrocolombians |
| `ARA` | Arawak |
| `BEJ` | Beijing |
| `HAD` | Hadza |
| `JEN` | Jenu Kurubas |
| `KRA` | Krakow |
| `LIM` | Rural Poland |
| `MBE` | Mbendjele BaYaka |
| `MEN` | Mentawai Islanders |
| `MES` | Colombian Mestizos |
| `NYA` | Nyangatom |
| `PNG` | Enga |
| `QUE` | Quechua |
| `SPA` | Sápara and Achuar |
| `TOP` | Toposa |
| `TSI` | Tsimane |
| `TUR` | Turku |
| `USD` | San Diego |
| `VAN` | Tannese Vanuatuans |
| `WEL` | Wellington |
| `TOR` | Toronto |

Correction note: earlier project prose expanded `MEN` as “Menominee” and `MES`
as “Mescalero.” The official meanings are **Mentawai Islanders** and
**Colombian Mestizos**. See the retained
[correction record](evidence/2026-07-11-fieldsite-name-correction.json).

### Other named datasets

| Name | Meaning and use here |
|---|---|
| **Music Lab human vocalization corpus** | The 21-society speech/song corpus used for Stage 2 and some natural-speech timing and prediction experiments. |
| **ESC-50** | Environmental Sound Classification dataset with 50 classes. Selected folds provide diffuse and impulsive non-vocal controls. |
| **VocalSet** | Recorded singer corpus used for held-out five-vowel perception tests. |
| **CMU Arctic** | Carnegie Mellon University speech corpus used for ordinary read-speech vowel development. |

## Evaluation metrics

For the formulas below:

- **TP**, true positive: IDS correctly predicted as IDS;
- **FP**, false positive: a negative example incorrectly predicted as IDS;
- **TN**, true negative: a negative example correctly rejected;
- **FN**, false negative: IDS incorrectly rejected.

| Metric | Meaning | Formula or interpretation |
|---|---|---|
| **Accuracy** | Fraction of all decisions that are correct | `(TP + TN) / total` |
| **Precision** | When the model predicts IDS, how often it is right | `TP / (TP + FP)` |
| **Recall** or **TPR** | Fraction of real IDS examples recovered | `TP / (TP + FN)` |
| **F1 score** | Harmonic mean of precision and recall | `2TP / (2TP + FP + FN)` |
| **FPR** | Fraction of actual negatives incorrectly promoted | `FP / (FP + TN)` |
| **Specificity** or **TNR** | Fraction of actual negatives correctly rejected | `TN / (TN + FP)`; equals `1 - FPR` |
| **AUC** | Area under a ranking curve | `0.5` is chance ranking; `1.0` is perfect ranking. It does not select an operating threshold. |
| **MAE** | Mean absolute error | Average absolute difference between prediction and target. |
| **RMSE** | Root mean squared error | Square root of average squared error; penalizes large errors more than MAE. |
| **p50** | Median | Half the observations are at or below it. |
| **p95** | 95th percentile | Used heavily for latency because a mean can hide slow frames. |
| **p99** | 99th percentile | A more tail-sensitive percentile than p95. |

An F1 improvement is not enough by itself. Stage 2 also limits overall FPR,
song FPR, and the minimum F1 of each fieldsite so pooled averages cannot hide a
failed society.

## Common Stage 2 scores and fields

| Field | Meaning |
|---|---|
| `vocal_score` | YIN voiced-window fraction. The current vocal front gate passes at `0.20` or above. |
| `predicted_vocal` | Whether the upstream vocal-activity gate passed. |
| `speech_score` | Evidence for speech rather than song. In predictive models, it is the best speech likelihood minus the best song likelihood. |
| `ids_score` | Evidence for IDS rather than ADS. In predictive models, it is IDS likelihood minus ADS likelihood. |
| `predicted_ids` | Final Boolean decision after calibration, vocal, speech/song, and IDS/ADS gates. |
| `calibrated` | Whether the example had the required unlabeled comparison observations for its speaker and vocal mode. |
| `calibration_observation_count` | Number of observations used in that within-speaker/mode calibration group. |
| `condition_mean_log_likelihood` | Average predictive log-likelihood under each condition model. Less-negative or larger values indicate a better model fit. |
| `decision_thresholds` | Frozen minimum score values used to turn continuous scores into decisions. |
| `song_false_positive_count` | Song clips incorrectly promoted as IDS. |
| `minimum_fieldsite_f1` | Worst F1 among evaluated fieldsites. |

## Audio and acoustics

| Term | Expansion | Meaning in this project |
|---|---|---|
| **ADC** | Analog-to-digital converter | Converts microphone voltage into digital samples. “ADC frame end” is the estimated device-side end time of a captured frame. |
| **dB** | Decibel | Logarithmic ratio. Cora’s digital amplitude values are not calibrated sound-pressure levels unless a protocol explicitly says so. |
| **dBFS** | Decibels relative to full scale | `0 dBFS` is the largest representable digital level. Negative values are quieter. |
| **SPL** | Sound pressure level | Calibrated physical acoustic level. Current speaker-knob experiments do **not** measure SPL. |
| **F0** | Fundamental frequency | Pitch-related repetition rate, in hertz. |
| **F1/F2/F3** | First/second/third formant | Vocal-tract resonance estimates, in hertz. Not the F1 classifier metric. |
| **FFT** | Fast Fourier transform | Converts a time-domain frame into frequency components. |
| **FIR** | Finite impulse response | A bounded filter represented by a fixed number of coefficients. The first natural-speech workspace candidate estimated a 128-sample self-path FIR and failed its live development-gap gate. |
| **Hz / kHz** | Hertz / kilohertz | Cycles per second / thousands of cycles per second. |
| **LPC** | Linear predictive coding | Estimates a spectral envelope and candidate formants from past samples. Used in the vowel perception route. |
| **PCM** | Pulse-code modulation | The decoded sequence of audio sample amplitudes. |
| **RMS** | Root mean square | Energy-like amplitude summary. Cora’s peak-RMS gate is a baseline, not a parentese detector. |
| **SNR** | Signal-to-noise ratio | Signal strength relative to noise, usually reported in dB. |
| **VAD** | Voice activity detection | Detecting whether voice is present. Cora’s supported five-second vocal front gate is narrower than general VAD. |
| **WAV** | Waveform Audio File Format | Container used for corpus and captured PCM audio. Cora reads integer PCM and IEEE-float WAV. |
| **YIN** | Name of a pitch-estimation algorithm, not an acronym used here | Supplies pitch and voiced-window coverage for the vocal gate. |
| **Spectral centroid** | — | Frequency-weighted center of spectral energy; loosely related to brightness. |
| **Spectral flux** | — | Amount of spectral change from one frame to the next. |
| **Hann window** | — | A smooth taper applied before an FFT to reduce edge discontinuities. The final natural-speech workspace metric applies one to each 600 ms capture. |
| **Spectral-shape distance** | — | Euclidean distance between mean-centered, L2-normalized log-band energy vectors. In spectral-v3 it compares a current event with the stored sensory trace for the matching efference identity. |
| **Clipping** | — | Samples reaching full digital scale. Clipping means information may have been lost. |
| **Frame** | — | A short block of audio analyzed as one observation, commonly 25 or 50 ms here. |
| **Sample rate** | — | Samples captured per second, commonly 16,000 Hz on the current XVF3800 path. |

### Units seen in reports

| Unit | Meaning |
|---|---|
| `s` | seconds |
| `ms` | milliseconds; one thousandth of a second |
| `Hz` | cycles per second |
| `kHz` | thousands of cycles per second |
| `dB` | logarithmic ratio; the reference must be stated or inferred from the protocol |
| `MiB` | mebibytes, `1,048,576` bytes; used for memory reporting |
| `J` | joules, a unit of energy |
| `µJ/frame` | microjoules per processed frame; one millionth of a joule per frame |

## Learning and model terms

| Term | Meaning in Cora-Infans |
|---|---|
| **Baseline** | A simpler comparator that the candidate must beat or at least be interpreted against. |
| **Batch calibration** | Uses a bounded group of unlabeled observations together before scoring them. This is not cold-start operation. |
| **Causal** | Uses only information available at or before the decision time. A model can use causal frame contexts but still be non-causal overall if its normalization uses a future batch. |
| **Codebook** | A bounded collection of learned prototype states. An observation maps to its nearest or most active prototype. |
| **Cohort adaptation** | Adjusts a model using unlabeled observations from a new multi-speaker community or fieldsite. |
| **Cold start** | Must work on the first observation from a new speaker/site without prior target examples. Current register models do not meet this boundary. |
| **CORAL** (Correlation Alignment) | A domain-adaptation method that aligns source and target covariance statistics. Cora maps an unlabeled target cohort into source geometry. |
| **Gaussian likelihood** | A score based on how probable an observed error is under an assumed bell-shaped residual distribution. |
| **Linear model** | Output changes as a weighted sum of inputs. “Linear” does not mean trivial; it describes the mathematical form. |
| **Nonlinear model** | Can represent relationships that are not a weighted straight-line combination of inputs. |
| **Prototype** | One representative point in a codebook or locally tuned model. |
| **Quantization error** | Distance from each observation to its nearest prototype. Lower means the codebook represents its training surface more closely. |
| **RBF** (radial basis function) | A locally tuned nonlinear activation based on distance from a prototype. |
| **Residual** | Difference between what was observed and what was predicted. |
| **Ridge regression** | Linear fitting with an L2 penalty that discourages excessively large coefficients. |
| **L2 norm / normalization** | Euclidean vector length / division by that length. Spectral-v3 uses it to remove overall gain while retaining relative 12-band shape. |
| **Transductive** | Uses unlabeled examples from the specific target batch when adapting or normalizing. |
| **Winner-local update** | Only the nearest prototype changes for a training observation. It is a local competitive-learning rule, not global backpropagation. |
| **Backpropagation** | Gradient-based credit assignment through multiple model layers. The bounded local codebook candidate did not use it. |

## Predictive and workspace architecture

| Term | Meaning |
|---|---|
| **Acoustic saliency** | A score for which sounds deserve processing or attention. Loudness is only one possible cue. |
| **Analysis by synthesis** | Perceive a sound by finding a motor/acoustic representation that could reproduce it. |
| **Bounded workspace** | Capacity-limited memory for active events. “Bounded” means memory cannot grow without limit. |
| **Corollary discharge** | An internal signal about a self-generated action, used to predict its sensory consequence. |
| **Efference copy** | A copy of a motor command supplied to prediction. In this project it marks known self-generated audio. |
| **Exact efference** | The same commanded waveform and identity used in an earlier self-calibration replay. The supported natural-speech H3 result depends on this narrow condition; it is not generalized self-voice learning. |
| **Sensory template** | A bounded stored consequence of an earlier action. Spectral-v3 stores 12 normalized band-energy scalars per active self identity, with three workspace slots. |
| **Dropout** | Deliberately missing observations used to test representation stability. This is not neural-network dropout unless explicitly stated. |
| **Local predictive learning** | Updates predictions using nearby observations and errors without a large globally trained sequence model. |
| **Motor plan** | Parameters used to synthesize a target sound, currently vowel source/formant settings. |
| **Parentese** | Informal term for infant-directed vocal style. It is culturally and relationally variable; Cora does not assume one universal acoustic template. |
| **Predictive coding** | Compare incoming observations with predictions and propagate the mismatch or residual. The term is a design hypothesis, not proof of neural fidelity. |
| **Self-suppression** | Reduce attention to predicted self-generated sound so unexpected world sound remains visible. |
| **Theta/gamma scheduling** | Project timing abstraction with slower event windows and faster subslots. It is not a measurement of brain rhythms. |
| **Top-down prediction** | A higher or slower state predicts what a lower level should observe next. |
| **Bottom-up residual** | Prediction error passed from observations toward a higher level. |
| **DIVA** (Directions Into Velocities of Articulators) | A speech motor-control model referenced as inspiration. Cora does not implement DIVA. |

## Hypotheses and evidence stages

The hypothesis number and stage number are separate systems.

| Code | Hypothesis |
|---|---|
| `H1` | Acoustic saliency: useful speech structure should improve attention beyond raw loudness. |
| `H2` | Local predictive learning: bounded local updates should improve held-out prediction. |
| `H3` | Bounded event workspace: capacity and scheduling should preserve bindings while reducing interference. |
| `H4` | Analysis by synthesis: perception and production should share a useful representation. |
| `H5` | Efficient embodied operation: the system should run with measured bounded latency and compute. |

| Stage | Evidence area |
|---|---|
| `0` | Reproducible software and explicit real/synthetic contracts |
| `1` | Real-audio decoding and evaluation |
| `2` | Parentese-relevant saliency |
| `3` | Predictive hierarchy |
| `4` | Workspace binding |
| `5` | Perception-production closure |
| `6` | Real-time efficiency |

## Hardware and runtime

| Term | Expansion | Meaning in Cora-Infans |
|---|---|---|
| **XVF3800** | XMOS XVF3800 voice processor | DSP chip on the Seeed Studio ReSpeaker microphone array. |
| **XMOS** | Vendor/platform name | Runs the microphone board’s on-device DSP. It is not an acronym expanded by this project. |
| **DSP** | Digital signal processing | Low-level audio processing; much of the microphone-array work happens on the XVF3800 before the host receives samples. |
| **USB** | Universal Serial Bus | Carries the board’s fused audio stream to the Mac. |
| **DoA** | Direction of arrival | Estimated source bearing. Host-side two-channel imbalance is only a coarse cue; firmware bearing is used only when finite and speech-classified. |
| **AEC** | Acoustic echo cancellation | Processing intended to remove known loudspeaker echo from microphone input. |
| **AGC** | Automatic gain control | Automatically changes input gain to stabilize level. |
| **HPF** | High-pass filter | Attenuates low-frequency content below a cutoff. |
| **CoreAudio** | Apple audio subsystem | macOS device layer used for playback and capture. |
| **PortAudio** | Cross-platform audio I/O library | Accessed through Python `sounddevice`; its callback timestamps support current latency measurements. |
| **Callback** | — | Function invoked by the audio system when a new sample block arrives. Cora reblocks variable callbacks into exact analysis frames. |
| **CPU** | Central processing unit | Reports process compute usage; percentage semantics depend on the host OS. |
| **RSS** | Resident set size | Process memory currently resident in physical RAM; reports use peak RSS. |
| **RTF** | Real-time factor | Processing wall time divided by represented audio duration. Below `1.0` means faster than real time. |
| **ADC-to-workspace latency** | — | Time from device-reported ADC frame timing to Cora’s workspace decision. It excludes unknown XVF3800 internal DSP delay. |
| **Deadline miss** | — | A frame whose measured processing boundary exceeded its frozen deadline. |
| **Overflow** | — | Input samples were lost because capture could not keep up. |
| **Fallback** | — | A secondary I/O path was used after the requested one failed. Real-evidence gates generally require no fallback. |
| **Device `:0`** | — | Current macOS/ffmpeg-style selector for the XVF3800 input. It is host-specific, not a universal device ID. |
| **Device `2`** | — | Current external-world playback output. Device indexes can change when hardware changes. |

## Evidence and data-governance words

| Term | Meaning in this repository |
|---|---|
| **Ablation** | Remove or replace one component to learn whether it caused the observed result. |
| **Artifact** | Retained file containing a protocol, model, manifest, result, audit, or correction. |
| **Claim boundary** | Explicit statement of what a result does and does not support. |
| **Confirmation** | Evaluation performed only after the model, thresholds, gates, and stop rule are frozen. |
| **Development data** | Data allowed for model design, fitting, or threshold selection. It cannot later become pristine confirmation evidence for that workflow. |
| **Descriptive result** | Measured observation that did not satisfy or was not eligible for a promotion gate. |
| **DOI** (Digital Object Identifier) | Persistent identifier for a paper or dataset. |
| **Falsified** | The frozen candidate failed its stated empirical criterion. It does not mean every imaginable version of the broad idea is impossible. |
| **Fresh / pristine** | Not previously inspected or used at the protected unit, such as participant or fieldsite, for the relevant workflow. |
| **Hash-pinned** | Exact file bytes are identified by a cryptographic digest so the input can be verified later. |
| **Held out** | Excluded from fitting or threshold selection and used only for evaluation. |
| **Manifest** | Inventory of selected files and provenance, often including hashes, labels, and split assignments. |
| **Participant-disjoint** | No participant appears in both training/development and evaluation groups. |
| **Preregistration** | Protocol and acceptance rules written before protected outcomes are inspected. |
| **Promotion** | Project status change allowed only when every frozen gate passes. |
| **Protected / reserved** | Data deliberately left untouched until a credible preregistered confirmation. |
| **Replication** | Repeat a frozen mechanism on a new participant, fieldsite, or condition. |
| **SHA-256** (Secure Hash Algorithm, 256 bit) | Digest used to verify exact artifact or audio bytes. It identifies bytes; it does not grant a license. |
| **Spent** | Already used or inspected at the relevant provenance boundary. Spent data remains useful for development and post-hoc analysis but not fresh confirmation. |
| **Synthetic / fake I/O** | Generated inputs used to test plumbing; cannot by itself support a real-hardware claim. |
| **Live / real I/O** | Actual device playback or capture occurred and the artifact records its provenance. |
| **CC BY** (Creative Commons Attribution) | Reuse requires attribution. Exact version and other conditions still matter. |
| **CC BY-NC-SA** (Creative Commons Attribution–NonCommercial–ShareAlike) | Requires attribution, restricts commercial use, and imposes share-alike terms. Check the source’s exact version. |

## Common artifact and JSON fields

| Field | Meaning |
|---|---|
| `experiment` / `mode` | Name of the protocol, model, or report type. |
| `status` / `verdict` | Current evidence outcome, not merely whether code ran. |
| `clip_id` | Filename stem identifying fieldsite, participant, and condition. |
| `speaker_id` | Fieldsite plus participant number, such as `KRA01`. |
| `fieldsite` | Three-letter corpus code. |
| `label` | Ground-truth condition used for evaluation. Labels must not generate evaluated audio features. |
| `split` | Usually `dev` or `held_out`. |
| `sample_rate` | Audio samples per second. |
| `frame_count` | Number of analyzed frames. Check `frame_ms` to know represented duration. |
| `salient_frame_count` | Frames that emitted an event after the saliency gate. |
| `prediction_count` | Number of scored examples or transitions, depending on the report. |
| `per_fieldsite` | Metrics reported separately for each held-out fieldsite. |
| `input_manifest_sha256` | Hash of the manifest bytes used by the run. |
| `all_clip_hashes_verified` | Whether every selected audio file matched its retained hash. |
| `confirmation_audio_or_metadata_accessed` | Must remain `false` during development-only work. |
| `development_candidate_ready` | Whether every frozen development gate passed and confirmation planning is credible. |
| `promotion_allowed` | Explicit guard against interpreting a failed or ineligible result as validated. |
| `real_capture` | Whether data came from a real capture adapter rather than a fake fixture. |
| `capture_backend` | Actual audio path used, such as the CoreAudio callback route. |
| `capture_stream_fallback_reason` | Why a secondary capture route was used; `null` means no fallback. |
| `clipped_sample_count` | Count of full-scale samples. Zero is required by current clean-capture gates. |
| `steady_state_real_time_factor` | RTF after startup/transient handling defined by that protocol. |
| `peak_resident_memory_megabytes` | Maximum process RSS in MiB-like report units. |

## File formats and tooling

| Term | Meaning |
|---|---|
| **CLI** | Command-line interface, invoked as `python -m cora_infans ...` or `cora-infans ...`. |
| **CSV** | Comma-separated values, used for corpus labels and manifests. |
| **JSON** | JavaScript Object Notation, the primary retained evidence format. |
| **RIFF** | Resource Interchange File Format, the chunked container underlying WAV. |
| **IEEE float WAV** | WAV whose samples use an IEEE floating-point representation rather than integer PCM. |
| **ZIP / ZIP64** | Archive formats; ZIP64 extends ZIP for large files and offsets. |

## Related documents

- [VISION.md](../VISION.md): hypotheses, evidence ladder, and claim boundaries.
- [Current project status](PROJECT_STATUS_2026-07-11.md): concise evidence disposition.
- [External research data policy](DATA_POLICY.md): licensing, provenance, and protected-data rules.
- [Evidence pipeline](architecture/evidence-pipeline.md): how observations become retained evidence.
