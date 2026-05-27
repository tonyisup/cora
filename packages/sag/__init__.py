"""Semantic Abstraction Graph — Corollary's internal knowledge representation.

The SAG is a directed, labeled graph where nodes represent semantic concepts
(entities, states, abstractions) and edges represent typed relationships.
It serves as the PFC's working memory backbone and a shared vocabulary
across all internal modules.
"""

from sag.acoustic_models import (
    AcousticNode,
    precedes_edge,
    similar_to_edge,
    sound_of_edge,
)
from sag.graph import SemanticAbstractionGraph
from sag.models import Edge, EdgeAssertion, GraphChange, Node

__all__ = [
    "SemanticAbstractionGraph",
    "Node",
    "Edge",
    "EdgeAssertion",
    "GraphChange",
    "AcousticNode",
    "sound_of_edge",
    "similar_to_edge",
    "precedes_edge",
]
