# M9 — Sound-first motor learning package.

from m9_sound.controller import ControllerParams, NeuralController
from m9_sound.convergence import RepeatabilityStats, compute_repeatability_stats, refine_prototype, summarize_audio
from m9_sound.oscillator import OscillatorBank
from m9_sound.producer import AcousticProducer, ProductionResult
from m9_sound.similarity import AcousticSimilarity, SimilarityResult
from m9_sound.substrate import AcousticSubstrate
from m9_sound.terminal_confirmation import (
    CombinedReward,
    TerminalJudgement,
    WeakTerminalJudge,
    attach_label_if_passed,
    combine_reward,
)
from m9_sound.temporal import (
    DEFAULT_TEMPORAL_TONES,
    count_decoded_transitions,
    decode_temporal_sequence,
    frame_audio,
    generate_tone_sequence,
    load_temporal_demo_producer,
    render_param_sequence,
    run_temporal_smoke_report,
    write_pcm16_wav,
    write_temporal_smoke_wav,
)
from m9_sound.live_capture import run_live_temporal_capture_report
from m9_sound.text_association import TextAssociationStats, TextAssociationStore, WordSoundPair
from m9_sound.vowel_transitions import (
    DEFAULT_LOOKUP_CKPT,
    DEFAULT_SUBSTRATE_CKPT,
    FAMILY_DURATIONS_S,
    VOWEL_FAMILIES,
    build_vowel_transition_report,
    write_vowel_transition_wav,
)

__all__ = [
    "OscillatorBank",
    "AcousticSubstrate",
    "NeuralController",
    "ControllerParams",
    "AcousticSimilarity",
    "SimilarityResult",
    "AcousticProducer",
    "ProductionResult",
    "RepeatabilityStats",
    "compute_repeatability_stats",
    "refine_prototype",
    "summarize_audio",
    "TerminalJudgement",
    "CombinedReward",
    "WeakTerminalJudge",
    "combine_reward",
    "attach_label_if_passed",
    "DEFAULT_TEMPORAL_TONES",
    "count_decoded_transitions",
    "decode_temporal_sequence",
    "frame_audio",
    "generate_tone_sequence",
    "load_temporal_demo_producer",
    "render_param_sequence",
    "run_temporal_smoke_report",
    "write_pcm16_wav",
    "write_temporal_smoke_wav",
    "run_live_temporal_capture_report",
    "TextAssociationStats",
    "TextAssociationStore",
    "WordSoundPair",
    "DEFAULT_LOOKUP_CKPT",
    "DEFAULT_SUBSTRATE_CKPT",
    "FAMILY_DURATIONS_S",
    "VOWEL_FAMILIES",
    "build_vowel_transition_report",
    "write_vowel_transition_wav",
]
