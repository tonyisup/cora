from cora_infans.saliency import compute_saliency_reward



def test_compute_saliency_reward_returns_peak_minus_mean_contrast() -> None:
    reward = compute_saliency_reward([0.2, 0.9, 0.4, 0.1])

    assert reward == 0.5



def test_compute_saliency_reward_returns_zero_for_uniform_energy() -> None:
    reward = compute_saliency_reward([0.5, 0.5, 0.5, 0.5])

    assert reward == 0.0
