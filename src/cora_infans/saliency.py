from __future__ import annotations


def compute_saliency_reward(channel_energies: list[float]) -> float:
    if not channel_energies:
        return 0.0

    peak_energy = max(channel_energies)
    mean_energy = sum(channel_energies) / len(channel_energies)
    reward = peak_energy - mean_energy
    return max(reward, 0.0)
