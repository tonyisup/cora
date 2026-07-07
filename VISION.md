# Motivation
Simulating the Sensory Gating Loop in Early Auditory Development

## Project Motivation

How do human infants effortlessly map the complex, high-dimensional acoustic space of human speech long before they can talk? Traditional machine learning models treat all training data equally, requiring millions of uniform audio frames to discover structural patterns. Human biology, however, uses a highly optimized **sensory gating mechanism** powered by a real-time neurochemical reward loop.

During the first year of life, infants do not passively process the ambient soundscape. Instead, primitive subcortical structures (like the *inferior colliculus* and *locus coeruleus*) act as deterministic physical filters. They filter specifically for **"parentese" (infant-directed speech)**—characterized by exaggerated fundamental frequency ($f_0$) pitch sweeps, prolonged vowel durations, and specific rhythmic modulation.

When these hardcoded physical thresholds are met, the infant's brain triggers an automated neuromodulatory surge of **acetylcholine and dopamine**. This localized blast of neuroplasticity acts as a biological "save button," opening up the primary auditory cortex to heavily weight and encode the precise spatial geometries of those specific vocal formants ($F_1/F_2$).

```
[Ambient Audio] ──> [Hardcoded Feature Gate] ──> [Saliency Burst] ──> [Dynamic Plasticity Map Update]

```

## Core Objective

The goal of this project is to build an audio AI architecture that replicates this biological paradigm. By replacing generic, unweighted backpropagation with an **Acoustic Saliency Reward Loop**, the system will:

1. Parse raw audio streams through deterministic, low-level DSP filters ($f_0$ variance and formant elongation gates).
2. Calculate a dynamic, real-time saliency reward metric ($W_s$) based on infant-directed speech characteristics.
3. Use this reward metric to dynamically scale the network's learning rate—simulating the gating of neuromodulators to learn clean, highly defined native vowel spaces using a fraction of the computational footprint of traditional models.


# System Architecture Specification: Auditory Saliency Prototype
**Project State:** Conceptual Blueprint  
**Target Focus:** Real-time cognitive sensory gating, attention shifting, and spatial audio processing.

---

## 1. System Overview & Data Flow

The simulation utilizes a modular, frame-based pipeline architecture that segregates low-level digital signal processing (DSP) and hardware telemetry collection from high-level cognitive state management and reinforcement learning reward loops.

+-------------------------------------------------------------------------+
|                  ReSpeaker XMOS XVF3800 Hardware Layer                  |
+-------------------------------------------------------------------------+
|                                              |
v (USB Audio Class 2.0 / I2S)                  v (Control Pipe via pyusb / I2C)
+-----------------------------------------+   +------------------------------------+
| 2-Channel Processed Audio Frame         |   | Intercepted Telemetry              |
| (Target isolated via blind source sep)  |   | - Direction of Arrival (DoA Index) |
|                                         |   | - Voice Activity Detection (VAD)   |
+-----------------------------------------+   +------------------------------------+
|                                              |
+-----------------------+----------------------+
|
v
+-------------------------------------------------------------------------+
|                    Python Simulation Application Core                   |
+-------------------------------------------------------------------------+
|  [Ingestion & Feature Window]                                           |
|  - Process raw 20-50ms buffers                                          |
|  - Perform spectral/formant extraction and SNR calculations             |
|                                                                         |
|  [Stateful Event Memory Matrix]                                         |
|  - Clustered track management (Object permanence for pauses/breaths)     |
|                                                                         |
|  [Cognitive Saliency Reward Loop]                                       |
|  - Evaluate intent, track stability, and trigger top-down overrides    |
+-------------------------------------------------------------------------+

---

## 2. Hardware Layer Specifications

* **Selected Hardware:** ReSpeaker XMOS XVF3800 Development Board.
* **Geometry:** 4-microphone planar square matrix layout (symmetrical coordinates at approximately $\pm 33\text{ mm}$ around a central axis).
* **Core Advantage:** The orthogonal X and Y mic pairings physically eliminate the front-back mirror ambiguity inherent to standard 1D 2-microphone broadside arrays.
* **Delegated Processing:** Hardware offloads Acoustic Echo Cancellation (AEC), automatic beamforming, and foundational Voice Activity Detection (VAD). 
* **Control Interfacing:** The Python core interfaces with the hardware via `pyusb` to intercept raw control registers, pulling native Direction of Arrival (DoA) index tracking data ($0^\circ \text{ to } 359^\circ$) to analyze and potentially override hardware-selected focus zones.

---

## 3. Software & Simulation Architecture

### Technical Stack
* **Runtime:** Python 3.11+
* **Numeric Computing:** `NumPy` and `SciPy` for vectorized matrix operations, rolling calculations, and fast short-term spectral analysis.
* **Type-Safety & Schemas:** `Pydantic` or Python native `dataclasses` to enforce predictable structures on data frames and telemetry inputs.
* **Future ML Core:** `PyTorch` (reserved for modeling differentiable neural attention masks or actor-critic gating loops).

### Memory Buffer Paradigm: Hybrid Stateful Event Tracker
To support complex attention shifting and achieve cognitive object permanence, the memory layout uses a dual-layer approach:
1. **Lightweight Rolling Window (20ms – 50ms):** Used exclusively for continuous frame-by-frame data ingestion, signal-to-noise ratio (SNR) calculations, and formant profile isolation.
2. **Stateful Event History Matrix:** Extracted features are pushed to a collection of discrete `Tracked Source Objects`. If a speaker pauses to take a breath, their entry enters an inactive cooling state rather than being wiped out, preserving attention state continuity.

---

## 4. Multi-Modal Roadmap Constraints
* **Spatial Coordinate Agnosticism:** All computed audio trajectories are stored using standard $360^\circ$ spatial coordinates rather than internal board indices.
* **Vision Integration Slot:** The spatial ring is designed to accept overlapping visual bounding box coordinates (e.g., from an OpenCV face detection loop) to enable cross-modal validation (correlating acoustic origin with visual presence to drive a robust biological reward mechanism).

## Proposed Architecture - Schema

```
from datetime import datetime
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, conint, confloat

class AcousticTelemetry(BaseModel):
    """Raw snapshot of spatial hardware metrics parsed from a frame window."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    doa_angle: conint(ge=0, le=359) = Field(
        ..., description="Direction of Arrival index mapped directly to a 360-degree ring."
    )
    vad_active: bool = Field(
        ..., description="Hardware-level Voice Activity Detection flag."
    )
    raw_snr: float = Field(
        ..., description="Calculated signal-to-noise ratio in decibels (dB) for the current frame."
    )

class SpectralProfile(BaseModel):
    """Extracted acoustic markers used to uniquely finger-print an identity."""
    formant_frequencies: Tuple[float, float, float] = Field(
        ..., description="Tracked center frequencies of the first three formants (F1, F2, F3)."
    )
    harmonic_to_noise_ratio (HNR): float = Field(
        ..., description="Measures voice quality and periodic structure predictability."
    )
    spectral_centroid: float = Field(
        ..., description="The center of mass of the spectrum, capturing brightness/timbre."
    )

class TrackedAudioSource(BaseModel):
    """The permanent cognitive agent representing a distinct speaker in the environment."""
    source_id: str = Field(..., description="Unique UUID or hashing token for the tracked source.")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Spatial Tracking State
    current_azimuth: float = Field(..., ge=0.0, lt=360.0, description="Smoothed spatial angle.")
    spatial_velocity: float = Field(0.0, description="Angular velocity (degrees per second) to predict movement.")
    
    # Cognitive Attention Metrics
    saliency_score: float = Field(1.0, ge=0.0, description="Dynamically calculated priority metric for the reward loop.")
    is_active_speaker: bool = Field(True, description="False if the speaker is currently pausing/breathing.")
    confidence_rating: float = Field(0.1, ge=0.0, le=1.0, description="Tracking certainty scale before object promotion.")
    
    # Feature Anchors
    historical_telemetry: List[AcousticTelemetry] = Field(default_factory=list, exclude=True)
    fingerprint: Optional[SpectralProfile] = Field(None, description="Anchored sound profile used for re-identification.")
    
    # Multi-Modal Hooks (Reserved for Camera Module)
    visual_bounding_box: Optional[Tuple[int, int, int, int]] = Field(
        None, description="Coordinates (x, y, w, h) from OpenCV face-detection loop once integrated."
    )
```