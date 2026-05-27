"""Acoustic extensions for the Semantic Abstraction Graph.

M9 needs graph nodes that can represent sound prototypes, phoneme-like
clusters, and exemplar recordings. These nodes stay lightweight and
serializable: we store summaries and feature vectors, not long raw waveforms.

This module does not change the core SAG runtime. It provides acoustic
node/edge helpers that higher-level M9 code can add into the existing graph.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from sag.models import Edge, EdgeType, Node


class AcousticNode(Node):
    """A sound prototype or acoustic concept in the SAG.

    The node can represent:
    - a target sound prototype
    - a phoneme-like cluster
    - an exemplar recording
    - a learned sound/word association

    We keep storage compact by using summary statistics and centroids rather
    than raw waveform blobs.
    """

    # Reference to the source waveform or a compressed representation.
    # For M9a/b we keep this optional and lightweight.
    prototype_audio: list[float] = Field(default_factory=list)

    # Compact acoustic summary vector.
    mfcc_centroid: list[float] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)

    # Simple acoustic descriptors.
    pitch_mean_hz: Optional[float] = None
    duration_mean_s: Optional[float] = None
    loudness_mean_db: Optional[float] = None

    # Learning / usage stats.
    production_count: int = 0
    avg_similarity: float = 0.0
    similarity_history: list[float] = Field(default_factory=list)

    # Optional semantic label for later text association.
    label: Optional[str] = None

    # Helpful provenance when the node is created from a particular trial.
    source_trials: list[str] = Field(default_factory=list)
    feature_version: str = "m9a-v1"

    model_config = {"frozen": True}

    def with_update(
        self,
        *,
        similarity: Optional[float] = None,
        trial_id: Optional[str] = None,
        label: Optional[str] = None,
        pitch_mean_hz: Optional[float] = None,
        duration_mean_s: Optional[float] = None,
        loudness_mean_db: Optional[float] = None,
        mfcc_centroid: Optional[list[float]] = None,
        embedding: Optional[list[float]] = None,
        prototype_audio: Optional[list[float]] = None,
    ) -> "AcousticNode":
        """Return a copy with incrementally updated summary stats.

        Because Node is frozen, updates happen by creating a new node version.
        M9b can use this to evolve prototype summaries over trials.
        """
        history = list(self.similarity_history)
        production_count = self.production_count
        avg_similarity = self.avg_similarity
        if similarity is not None:
            history.append(float(similarity))
            production_count += 1
            avg_similarity = sum(history) / len(history)

        trials = list(self.source_trials)
        if trial_id is not None and trial_id not in trials:
            trials.append(trial_id)

        return AcousticNode(
            id=self.id,
            core=self.core,
            context=dict(self.context),
            grounding=list(self.grounding),
            prototype_audio=list(prototype_audio if prototype_audio is not None else self.prototype_audio),
            mfcc_centroid=list(mfcc_centroid if mfcc_centroid is not None else self.mfcc_centroid),
            embedding=list(embedding if embedding is not None else self.embedding),
            pitch_mean_hz=self.pitch_mean_hz if pitch_mean_hz is None else pitch_mean_hz,
            duration_mean_s=self.duration_mean_s if duration_mean_s is None else duration_mean_s,
            loudness_mean_db=self.loudness_mean_db if loudness_mean_db is None else loudness_mean_db,
            production_count=production_count,
            avg_similarity=avg_similarity,
            similarity_history=history,
            label=self.label if label is None else label,
            source_trials=trials,
            feature_version=self.feature_version,
        )

    @classmethod
    def from_audio_summary(
        cls,
        *,
        id: str,
        core: str,
        grounding: Optional[list[str]] = None,
        context: Optional[dict[str, Any]] = None,
        label: Optional[str] = None,
        prototype_audio: Optional[list[float]] = None,
        mfcc_centroid: Optional[list[float]] = None,
        embedding: Optional[list[float]] = None,
        pitch_mean_hz: Optional[float] = None,
        duration_mean_s: Optional[float] = None,
        loudness_mean_db: Optional[float] = None,
        source_trials: Optional[list[str]] = None,
        feature_version: str = "m9a-v1",
    ) -> "AcousticNode":
        """Construct an acoustic node from summary values."""
        return cls(
            id=id,
            core=core,
            grounding=grounding or [],
            context=context or {},
            label=label,
            prototype_audio=prototype_audio or [],
            mfcc_centroid=mfcc_centroid or [],
            embedding=embedding or [],
            pitch_mean_hz=pitch_mean_hz,
            duration_mean_s=duration_mean_s,
            loudness_mean_db=loudness_mean_db,
            production_count=1 if prototype_audio or mfcc_centroid or embedding else 0,
            avg_similarity=0.0,
            similarity_history=[],
            source_trials=source_trials or [],
            feature_version=feature_version,
        )


def sound_of_edge(source: str, target: str, confidence: float = 1.0) -> Edge:
    """Create a SOUND_OF edge (sound -> word/concept)."""
    return Edge(source=source, target=target, type=EdgeType.SOUND_OF.value, confidence=confidence)


def similar_to_edge(source: str, target: str, confidence: float = 1.0) -> Edge:
    """Create a SIMILAR_TO edge between acoustic nodes."""
    return Edge(source=source, target=target, type=EdgeType.SIMILAR_TO.value, confidence=confidence)


def precedes_edge(source: str, target: str, confidence: float = 1.0) -> Edge:
    """Create a PRECEDES edge for temporal ordering."""
    return Edge(source=source, target=target, type=EdgeType.PRECEDES.value, confidence=confidence)
