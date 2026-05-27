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
]
