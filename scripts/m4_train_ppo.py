#!/usr/bin/env python3
"""M4 PPO training on the shaped speaking-policy environment.

This trainer starts from the distilled M3 checkpoint and performs a small
PPO-style update loop over the shaped-reward scenario environment.

Important design choice:
- episodes are one-step scenario rollouts, so this is intentionally a
  conservative PPO / actor-critic setup rather than a full long-horizon RL
  system. That keeps the learning loop stable and auditable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch is required. Run: .venv/bin/pip install torch")

from m2_train import ACTION_INDEX, ATTR_DIM, NUM_ACTIONS, STATE_DIM, DistilledSpeakingPolicy, encode_features
from m3_scenario_benchmark import SCENARIO_SUITE
from m4_rollout_env import RewardedScenarioEnv


ACTION_NAMES = ["speak", "wait", "clarify", "repair", "yield"]


@dataclass(frozen=True)
class Transition:
    features: torch.Tensor
    action: int
    log_prob: torch.Tensor
    reward: float
    value: torch.Tensor
    advantage: float = 0.0
    return_: float = 0.0


class ActorCritic(nn.Module):
    """Actor-critic network initialized from the distilled policy."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        input_dim = STATE_DIM + ATTR_DIM + 5
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def predict(self, features: Dict[str, Any]) -> str:
        x = torch.tensor(encode_features(features), dtype=torch.float32).unsqueeze(0)
        logits, _ = self.forward(x)
        return ACTION_NAMES[int(torch.argmax(logits, dim=-1).item())]


def _load_distilled_weights(actor_critic: ActorCritic, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    distilled = DistilledSpeakingPolicy(hidden_dim=64, dropout=0.1)
    distilled.load_state_dict(ckpt["model_state"])

    # Copy matching trunk weights.
    distilled_sd = distilled.state_dict()
    ac_sd = actor_critic.state_dict()
    mapping = {
        "backbone.0.weight": "net.0.weight",
        "backbone.0.bias": "net.0.bias",
        "backbone.3.weight": "net.3.weight",
        "backbone.3.bias": "net.3.bias",
    }
    for dst, src in mapping.items():
        ac_sd[dst] = distilled_sd[src].clone()

    # Initialize the policy head from the last distilled layer.
    ac_sd["policy_head.weight"] = distilled_sd["net.6.weight"].clone()
    ac_sd["policy_head.bias"] = distilled_sd["net.6.bias"].clone()
    # Value head remains random; that’s expected.
    actor_critic.load_state_dict(ac_sd)
    return ckpt


def _scenario_to_tensor(features: Dict[str, Any]) -> torch.Tensor:
    return torch.tensor(encode_features(features), dtype=torch.float32)


def collect_rollouts(policy: ActorCritic, env: RewardedScenarioEnv, device: torch.device, episodes: int) -> List[Transition]:
    transitions: List[Transition] = []
    policy.eval()
    order = list(range(len(env.scenarios)))
    random.shuffle(order)
    idx = 0

    for _ in range(episodes):
        if idx >= len(order):
            random.shuffle(order)
            idx = 0
        scenario_idx = order[idx]
        idx += 1
        obs = env.reset(scenario_idx)
        x = _scenario_to_tensor(obs).to(device)
        logits, value = policy(x.unsqueeze(0))
        logits = logits.squeeze(0)
        value = value.squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action, device=device))
        action_name = ACTION_NAMES[action]
        _, reward, done, _ = env.step(action_name)
        transitions.append(
            Transition(
                features=x.detach().cpu(),
                action=action,
                log_prob=log_prob.detach(),
                reward=float(reward),
                value=value.detach(),
            )
        )
    return transitions


def compute_advantages(transitions: List[Transition], gamma: float = 1.0) -> List[Transition]:
    # One-step episodes: return == reward. Still computed explicitly for clarity.
    rewards = [t.reward for t in transitions]
    returns = [r for r in rewards]
    values = [float(t.value.item()) for t in transitions]
    advs = [ret - val for ret, val in zip(returns, values)]
    adv_mean = float(np.mean(advs)) if advs else 0.0
    adv_std = float(np.std(advs)) if advs else 1.0
    adv_std = adv_std if adv_std > 1e-8 else 1.0

    out: List[Transition] = []
    for t, ret, adv in zip(transitions, returns, advs):
        out.append(
            Transition(
                features=t.features,
                action=t.action,
                log_prob=t.log_prob,
                reward=t.reward,
                value=t.value,
                advantage=(adv - adv_mean) / adv_std,
                return_=ret,
            )
        )
    return out


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    device: torch.device,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 32,
) -> Dict[str, float]:
    policy.train()
    feats = torch.stack([t.features for t in transitions]).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.stack([t.log_prob for t in transitions]).to(device)
    advantages = torch.tensor([t.advantage for t in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([t.return_ for t in transitions], dtype=torch.float32, device=device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_batches = 0

    indices = list(range(len(transitions)))
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_feats = feats[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            logits, values = policy(batch_feats)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, batch_returns)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_batches += 1

    return {
        "policy_loss": total_policy_loss / max(total_batches, 1),
        "value_loss": total_value_loss / max(total_batches, 1),
        "entropy": total_entropy / max(total_batches, 1),
    }


def evaluate_policy(policy: ActorCritic, env: RewardedScenarioEnv, device: torch.device) -> Dict[str, Any]:
    policy.eval()
    rows = []
    correct = 0
    total_reward = 0.0
    for i, scenario in enumerate(env.scenarios):
        obs = env.reset(i)
        x = _scenario_to_tensor(obs).to(device)
        with torch.no_grad():
            logits, _ = policy(x.unsqueeze(0))
            pred_idx = int(torch.argmax(logits, dim=-1).item())
        action = ACTION_NAMES[pred_idx]
        _, reward, _, info = env.step(action)
        ok = action == info["expected_action"]
        correct += int(ok)
        total_reward += reward
        rows.append(
            {
                "name": info["scenario_name"],
                "category": info["category"],
                "expected": info["expected_action"],
                "predicted": action,
                "reward": reward,
                "correct": ok,
            }
        )

    return {
        "accuracy": correct / len(env.scenarios),
        "total_reward": total_reward,
        "rows": rows,
    }


def save_checkpoint(path: Path, policy: ActorCritic, optimizer: torch.optim.Optimizer, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PPO on shaped speaking-policy rewards")
    parser.add_argument("--init-checkpoint", required=True, help="Distilled checkpoint to initialize from")
    parser.add_argument("--episodes", type=int, default=512, help="Number of rollout episodes per iteration")
    parser.add_argument("--iterations", type=int, default=20, help="Number of PPO iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/m4_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(hidden_dim=64, dropout=0.1).to(device)
    init_meta = _load_distilled_weights(policy, args.init_checkpoint, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    env = RewardedScenarioEnv()
    history: List[Dict[str, Any]] = []

    baseline_eval = evaluate_policy(policy, env, device)
    print(json.dumps({"baseline": baseline_eval}, indent=2))
    if args.eval_only:
        return 0

    best_reward = -math.inf
    best_path = None

    for it in range(1, args.iterations + 1):
        transitions = collect_rollouts(policy, env, device, args.episodes)
        transitions = compute_advantages(transitions)
        stats = ppo_update(
            policy,
            optimizer,
            transitions,
            device,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        eval_report = evaluate_policy(policy, env, device)
        record = {
            "iteration": it,
            "rollout_mean_reward": float(np.mean([t.reward for t in transitions])) if transitions else 0.0,
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "eval_accuracy": eval_report["accuracy"],
            "eval_total_reward": eval_report["total_reward"],
        }
        history.append(record)
        print(json.dumps(record))

        if eval_report["total_reward"] > best_reward:
            best_reward = eval_report["total_reward"]
            out_dir = Path(args.output_dir)
            best_path = out_dir / "best_policy.pt"
            save_checkpoint(best_path, policy, optimizer, {"init_meta": init_meta, "history": history, "best_eval": eval_report})

    final_path = Path(args.output_dir) / "final_policy.pt"
    save_checkpoint(final_path, policy, optimizer, {"init_meta": init_meta, "history": history})
    print(json.dumps({"final": evaluate_policy(policy, env, device), "best_path": str(best_path), "final_path": str(final_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
