"""M9d — Text association helpers.

This module connects stable sound prototypes to words using bidirectional SAG
edges. It stays synthetic for now: the goal is to prove that the word/sound
wiring and retrieval path are coherent before any live audio promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from m9_sound.convergence import summarize_audio
from m9_sound.oscillator import OscillatorBank
from sag import AcousticNode, SemanticAbstractionGraph, has_sound_edge, sound_of_edge
from sag.models import Node


@dataclass(frozen=True)
class WordSoundPair:
    word: str
    sound_id: str
    word_id: str
    freq_hz: float
    waveform: str


@dataclass(frozen=True)
class TextAssociationStats:
    total_pairs: int
    word_to_sound_hits: int
    sound_to_word_hits: int

    @property
    def word_to_sound_accuracy(self) -> float:
        return self.word_to_sound_hits / max(self.total_pairs, 1)

    @property
    def sound_to_word_accuracy(self) -> float:
        return self.sound_to_word_hits / max(self.total_pairs, 1)

    @property
    def all_passed(self) -> bool:
        return self.word_to_sound_hits == self.total_pairs and self.sound_to_word_hits == self.total_pairs


class TextAssociationStore:
    """Store and retrieve word-sound associations inside the SAG."""

    def __init__(self, graph: Optional[SemanticAbstractionGraph] = None, *, sr: int = 16000) -> None:
        self.graph = graph or SemanticAbstractionGraph()
        self._sr = sr
        self._osc = OscillatorBank(sr=sr)
        self._pairs: list[WordSoundPair] = []

    @property
    def pairs(self) -> list[WordSoundPair]:
        return list(self._pairs)

    def add_pair(
        self,
        word: str,
        *,
        freq_hz: float,
        waveform: str = "sine",
        duration_s: float = 1.0,
        confidence: float = 1.0,
    ) -> WordSoundPair:
        word_id = f"word::{word}"
        sound_id = f"sound::{word}"

        word_node = Node(
            id=word_id,
            core=f"Word node for {word}",
            grounding=[word],
            context={"kind": "word", "word": word},
        )
        self.graph.add_node(word_node)

        waveform_fn = {
            "sine": self._osc.sine,
            "triangle": self._osc.triangle,
            "sawtooth": self._osc.sawtooth,
            "square": self._osc.square,
        }[waveform]
        audio = waveform_fn(freq_hz, duration_s, amplitude=0.5)
        summary = summarize_audio(audio, self._sr)
        sound_node = AcousticNode.from_audio_summary(
            id=sound_id,
            core=f"Sound prototype for {word}",
            grounding=[word, waveform, f"{int(round(freq_hz))}hz"],
            context={"kind": "sound", "word": word, "freq_hz": freq_hz, "waveform": waveform},
            label=word,
            prototype_audio=audio.tolist(),
            mfcc_centroid=summary["mfcc_centroid"],
            pitch_mean_hz=summary["pitch_hz"],
            duration_mean_s=summary["duration_s"],
            loudness_mean_db=summary["loudness_db"],
            source_trials=[f"target::{word}"],
        )
        self.graph.add_node(sound_node)

        self.graph.add_edge(has_sound_edge(word_id, sound_id, confidence=confidence))
        self.graph.add_edge(sound_of_edge(sound_id, word_id, confidence=confidence))

        pair = WordSoundPair(
            word=word,
            sound_id=sound_id,
            word_id=word_id,
            freq_hz=freq_hz,
            waveform=waveform,
        )
        self._pairs.append(pair)
        return pair

    def add_pairs(self, items: Iterable[tuple[str, float, str]]) -> list[WordSoundPair]:
        return [self.add_pair(word, freq_hz=freq, waveform=waveform) for word, freq, waveform in items]

    def get_sound_for_word(self, word: str) -> AcousticNode | None:
        word_node = self.graph.resolve(word)
        if word_node is None:
            return None
        edge = self.graph.get_edge(word_node.id, f"sound::{word}", "has-sound")
        if edge is None:
            # fallback: traverse outgoing HAS_SOUND edges and return the first sound node
            for candidate in self.graph.neighbors(word_node.id, rel_types=["has-sound"]):
                node = self.graph.get_node(candidate.target)
                if isinstance(node, AcousticNode):
                    return node
            return None
        node = self.graph.get_node(edge.target)
        return node if isinstance(node, AcousticNode) else None

    def get_word_for_sound(self, sound_id: str) -> Node | None:
        sound_node = self.graph.get_node(sound_id)
        if sound_node is None:
            return None
        label = getattr(sound_node, "label", None)
        if isinstance(label, str) and label.strip():
            word_node = self.graph.get_node(f"word::{label}")
            if word_node is not None:
                return word_node
        # Fallback: traverse outgoing SOUND_OF edges to the linked word node.
        for candidate in self.graph.neighbors(sound_id, rel_types=["sound-of"]):
            return self.graph.get_node(candidate.target)
        return None

    def evaluate(self) -> TextAssociationStats:
        word_hits = 0
        sound_hits = 0
        for pair in self._pairs:
            sound = self.get_sound_for_word(pair.word)
            if sound is not None and sound.id == pair.sound_id:
                word_hits += 1
            word = self.get_word_for_sound(pair.sound_id)
            if word is not None and word.id == pair.word_id:
                sound_hits += 1
        return TextAssociationStats(
            total_pairs=len(self._pairs),
            word_to_sound_hits=word_hits,
            sound_to_word_hits=sound_hits,
        )
