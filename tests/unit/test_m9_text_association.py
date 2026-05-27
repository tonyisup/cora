"""Tests for M9d text association."""

from __future__ import annotations

from m9_sound import TextAssociationStore


class TestTextAssociationStore:
    def test_add_pair_and_retrieve_both_directions(self):
        store = TextAssociationStore()
        pair = store.add_pair("alpha", freq_hz=220.0, waveform="sine")
        sound = store.get_sound_for_word("alpha")
        word = store.get_word_for_sound(pair.sound_id)

        assert pair.word == "alpha"
        assert sound is not None
        assert sound.id == pair.sound_id
        assert word is not None
        assert word.id == pair.word_id

    def test_add_multiple_pairs_and_evaluate(self):
        store = TextAssociationStore()
        pairs = [
            ("alpha", 220.0, "sine"),
            ("beta", 247.0, "triangle"),
            ("gamma", 262.0, "sawtooth"),
        ]
        store.add_pairs(pairs)
        stats = store.evaluate()

        assert stats.total_pairs == 3
        assert stats.all_passed is True
        assert stats.word_to_sound_accuracy == 1.0
        assert stats.sound_to_word_accuracy == 1.0

    def test_graph_contains_bidirectional_edges(self):
        store = TextAssociationStore()
        pair = store.add_pair("delta", freq_hz=294.0, waveform="square")
        has_sound = store.graph.get_edge(pair.word_id, pair.sound_id, "has-sound")
        sound_of = store.graph.get_edge(pair.sound_id, pair.word_id, "sound-of")

        assert has_sound is not None
        assert sound_of is not None
        assert has_sound.source == pair.word_id
        assert sound_of.source == pair.sound_id
