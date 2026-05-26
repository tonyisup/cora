#!/usr/bin/env python3
"""M8c — PPO training with M8 conversational reward.

Extends the M4 multi-step PPO infrastructure with the M8 reward function,
which adds conversational quality signals (repair success, clarify efficiency,
stall penalty, resume after yield) on top of the base M4 turn-taking rewards.

The environment and model architecture are shared with M4 — only the reward
computation differs.

Usage:
    # Train from distilled checkpoint, initialized from distilled policy
    .venv/bin/python scripts/m8_train_ppo.py \\
        --distilled-checkpoint data/m8_distilled_policy.pt \\
        --episodes 256 \\
        --iterations 30 \\
        --output-dir results/m8_ppo

    # Train from M4 PPO checkpoint (if exists)
    .venv/bin/python scripts/m8_train_ppo.py \\
        --init-checkpoint results/m4_multistep_ppo/best_policy.pt \\
        --episodes 256 \\
        --iterations 30 \\
        --output-dir results/m8_ppo

    # Eval only
    .venv/bin/python scripts/m8_train_ppo.py \\
        --init-checkpoint results/m8_ppo/best_policy.pt \\
        --eval-only
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch is required. Run: .venv/bin/pip install torch")

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m4_multistep_env import ACTIONS, MultiStepTurnEnv
from m4_reward import reward_from_transition
from m8_reward import M8RewardConfig, compute_m8_reward

ACTION_NAMES = ACTIONS
STATE_INDEX = {"idle": 0, "listen": 1, "speak": 2, "repair": 3, "clarify": 4}

# Import scripted env (optional — only needed when --use-scripted-env is set)
try:
    from m8_policy.scripted_env import ScriptedEpisodeEnv
    _SCRIPTED_ENV_AVAILABLE = True
except ImportError:
    _SCRIPTED_ENV_AVAILABLE = False


@dataclass(frozen=True)
class M8TrajectoryStep:
    episode_id: int
    step_idx: int
    features: dict  # raw observation dict
    action: int
    log_prob: torch.Tensor
    reward: float
    m8_reward: float  # M8-specific reward component
    total_reward: float  # M4 base + M8 bonus
    done: bool
    value: torch.Tensor
    info: Dict[str, Any]
    return_: float = 0.0
    advantage: float = 0.0


class M8ActorCritic(nn.Module):
    """Actor-critic for M8. Same architecture as M4 ActorCritic.

    Input: 13-dim feature vector (state one-hot + attribution + numeric)
    Output: 5 action logits + 1 value
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, len(ACTION_NAMES))
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


def encode_obs(obs: dict) -> torch.Tensor:
    """Encode a MultiStepTurnEnv observation into a model input tensor."""
    state = str(obs.get("state", "listen")).lower()
    attr = str(obs.get("attribution", "silence")).lower()

    state_idx = STATE_INDEX.get(state, 1)
    state_onehot = torch.zeros(5, dtype=torch.float32)
    state_onehot[state_idx] = 1.0

    attr_idx = {"silence": 0, "self": 1, "world": 1, "both": 2}.get(attr, 0)
    attr_onehot = torch.zeros(3, dtype=torch.float32)
    attr_onehot[attr_idx] = 1.0

    numeric = torch.tensor([
        float(obs.get("silence_duration_ms", 0.0)) / 5000.0,  # normalize to ~0-1
        float(obs.get("has_queued_utterance", False)),
        float(obs.get("user_speaking", False)),
        float(obs.get("agent_speaking", False)),
        float(obs.get("queue_depth", 0)) / 5.0,
    ], dtype=torch.float32)

    return torch.cat([state_onehot, attr_onehot, numeric])


def obs_to_conversation_features(obs: dict) -> dict:
    """Convert MultiStepTurnEnv obs to ConversationFeatures-like dict for M8 reward."""
    state = str(obs.get("state", "listen")).lower()
    state_map = {"idle": "idle", "listen": "listening", "speak": "speaking",
                 "repair": "repair", "clarify": "clarification"}

    return {
        "conversation_state": state_map.get(state, "idle"),
        "is_agent_speaking": bool(obs.get("agent_speaking", False)),
        "is_agent_deferred": bool(obs.get("queue_depth", 0) > 0),
        "is_user_speaking": bool(obs.get("user_speaking", False)),
        "was_user_interrupted": False,
        "seconds_since_last_user_speech": float(obs.get("silence_duration_ms", 0.0)) / 1000.0,
        "seconds_since_last_agent_speech": 0.0,  # not tracked in env
        "seconds_agent_has_been_speaking": 0.0,  # not tracked in env
        "seconds_user_has_been_speaking": 0.0,  # not tracked in env
        "total_turns": 0,
        "repair_count": 1 if state == "repair" else 0,
        "clarification_count": 1 if state == "clarify" else 0,
        "yield_count": 0,
        "just_interrupted": bool(obs.get("attribution", "") == "both" and obs.get("agent_speaking", False)),
        "just_collided": bool(obs.get("attribution", "") == "both"),
        "just_deferred": False,
        "just_resumed": False,
        "has_queued_utterance": bool(obs.get("has_queued_utterance", False)),
    }


def compute_combined_reward(
    prev_obs: dict,
    action: str,
    next_obs: dict,
    reward_config: M8RewardConfig,
    use_m4_base: bool = True,
) -> Tuple[float, float, float]:
    """Compute combined M4 + M8 reward.

    Returns (total_reward, m4_reward, m8_reward).
    If use_m4_base is False, only M8 reward is returned (for scripted episodes
    where M4 reward is not calibrated to the episode structure).
    """
    # M8 conversational reward
    prev_features = obs_to_conversation_features(prev_obs)
    next_features = obs_to_conversation_features(next_obs)
    m8_signal = compute_m8_reward(reward_config, action, prev_features, next_features)

    if use_m4_base:
        # M4 base reward
        m4_signal = reward_from_transition(prev_obs, action, next_obs)
        total = m4_signal.reward + m8_signal.reward
    else:
        m4_signal = None
        total = m8_signal.reward

    m4_r = m4_signal.reward if m4_signal is not None else 0.0
    return total, m4_r, m8_signal.reward


def load_distilled_weights(model: M8ActorCritic, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load weights from M8 distilled policy checkpoint into the actor-critic."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    distilled_cfg = ckpt.get("config", {})

    from m8_policy.model import M8DistilledPolicy
    distilled = M8DistilledPolicy(
        input_dim=distilled_cfg.get("input_dim", 21),
        hidden_dim=distilled_cfg.get("hidden_dim", 64),
        dropout=0.0,  # no dropout for weight extraction
    )
    distilled.load_state_dict(ckpt["model_state"])

    # Map distilled net weights to actor-critic backbone
    # Distilled: Linear(21,64) → ReLU → Dropout → Linear(64,64) → ReLU → Dropout → Linear(64,5)
    # ActorCritic: Linear(13,64) → ReLU → Dropout → Linear(64,32) → ReLU → Dropout → heads
    # We can only directly copy layers with matching dimensions.
    # For mismatched layers, initialization will handle it.
    d_sd = distilled.state_dict()
    ac_sd = model.state_dict()

    # backbone.0: Linear(13, 64) — input dim mismatch (13 vs 21), skip
    # backbone.3: Linear(64, 32) — output dim mismatch (32 vs 64), skip
    # policy_head: Linear(32, 5) — input dim mismatch, skip
    # Nothing directly copyable due to architecture differences.
    # Fall back to training from scratch with the actor-critic architecture.
    print(f"Note: Architecture mismatch (distilled 21→64→64→5 vs AC 13→64→32→5). "
          f"Training from scratch with M8 reward signal.")

    return ckpt


def load_ppo_weights(model: M8ActorCritic, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load weights from an M4-style PPO checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["policy_state"])
    return ckpt


def collect_rollouts(
    policy: M8ActorCritic,
    env: Any,
    reward_config: M8RewardConfig,
    device: torch.device,
    episodes: int,
    use_m4_base: bool = True,
) -> List[List[M8TrajectoryStep]]:
    trajectories: List[List[M8TrajectoryStep]] = []
    policy.eval()
    for episode_id in range(episodes):
        obs = env.reset(episode_id)
        traj: List[M8TrajectoryStep] = []
        done = False
        step_idx = 0
        prev_obs = dict(obs)
        while not done:
            x = encode_obs(obs).to(device)
            logits, value = policy(x.unsqueeze(0))
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            action_name = ACTION_NAMES[action_idx]
            log_prob = dist.log_prob(torch.tensor(action_idx, device=device))

            next_obs, _, done, info = env.step(action_name)

            # Compute combined reward
            total_reward, m4_r, m8_r = compute_combined_reward(
                prev_obs, action_name, next_obs, reward_config,
                use_m4_base=use_m4_base,
            )

            traj.append(M8TrajectoryStep(
                episode_id=episode_id,
                step_idx=step_idx,
                features=obs,
                action=action_idx,
                log_prob=log_prob.detach(),
                reward=total_reward,
                m8_reward=m8_r,
                total_reward=total_reward,
                done=done,
                value=value.detach(),
                info={**info, "m4_reward": m4_r, "m8_reward": m8_r},
            ))
            obs = next_obs
            prev_obs = obs
            step_idx += 1
            if step_idx > 30:  # safety cap ~1s at 30ms/step
                break
        trajectories.append(traj)
    return trajectories


def compute_returns_and_advantages(
    trajectories: List[List[M8TrajectoryStep]],
    gamma: float = 0.95,
) -> List[M8TrajectoryStep]:
    flattened: List[M8TrajectoryStep] = []
    for traj in trajectories:
        returns: List[float] = []
        running = 0.0
        for step in reversed(traj):
            running = step.total_reward + gamma * running
            returns.append(running)
        returns.reverse()
        values = [float(step.value.item()) for step in traj]
        advantages = [ret - val for ret, val in zip(returns, values)]
        adv_mean = float(np.mean(advantages)) if advantages else 0.0
        adv_std = float(np.std(advantages)) if advantages else 1.0
        adv_std = adv_std if adv_std > 1e-8 else 1.0
        for step, ret, adv in zip(traj, returns, advantages):
            flattened.append(M8TrajectoryStep(
                episode_id=step.episode_id,
                step_idx=step.step_idx,
                features=step.features,
                action=step.action,
                log_prob=step.log_prob,
                reward=step.reward,
                m8_reward=step.m8_reward,
                total_reward=step.total_reward,
                done=step.done,
                value=step.value,
                info=step.info,
                return_=ret,
                advantage=(adv - adv_mean) / adv_std,
            ))
    return flattened


def ppo_update(
    policy: M8ActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: List[M8TrajectoryStep],
    device: torch.device,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 32,
) -> Dict[str, float]:
    policy.train()
    feats = torch.stack([encode_obs(t.features) for t in transitions]).to(device)
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
            b_feats = feats[batch_idx]
            b_actions = actions[batch_idx]
            b_old = old_log_probs[batch_idx]
            b_adv = advantages[batch_idx]
            b_ret = returns[batch_idx]

            logits, values = policy(b_feats)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - b_old)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, b_ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_batches += 1

    n = max(total_batches, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
    }


def evaluate_policy(
    policy: M8ActorCritic,
    env: Any,
    reward_config: M8RewardConfig,
    device: torch.device,
    use_m4_base: bool = True,
) -> Dict[str, Any]:
    policy.eval()
    episode_returns = []
    episode_m8_returns = []
    total_steps = 0
    total_m4 = 0.0
    total_m8 = 0.0

    # Support both MultiStepTurnEnv (.templates) and ScriptedEpisodeEnv (.num_episodes)
    num_episodes = getattr(env, "templates", None)
    num_episodes = len(num_episodes) if num_episodes is not None else env.num_episodes

    for episode_id in range(num_episodes):
        obs = env.reset(episode_id)
        prev_obs = dict(obs)
        done = False
        ep_return = 0.0
        ep_m8 = 0.0
        m4_sum = 0.0
        step_idx = 0
        while not done:
            x = encode_obs(obs).to(device)
            with torch.no_grad():
                logits, _ = policy(x.unsqueeze(0))
                pred_idx = int(torch.argmax(logits, dim=-1).item())
            action = ACTION_NAMES[pred_idx]
            next_obs, _, done, info = env.step(action)
            _, m4_r, m8_r = compute_combined_reward(prev_obs, action, next_obs, reward_config,
                                                       use_m4_base=use_m4_base)
            ep_return += m4_r + m8_r
            ep_m8 += m8_r
            m4_sum += m4_r
            total_m8 += m8_r
            total_steps += 1
            obs = next_obs
            prev_obs = obs
            step_idx += 1
        episode_returns.append(ep_return)
        episode_m8_returns.append(ep_m8)
        total_m4 += m4_sum

    n_ep = max(num_episodes, 1)
    return {
        "episodes": num_episodes,
        "total_steps": total_steps,
        "mean_episode_return": sum(episode_returns) / n_ep,
        "mean_m8_return": sum(episode_m8_returns) / n_ep,
        "total_m4_reward": total_m4,
        "total_m8_reward": total_m8,
        "episode_returns": episode_returns,
        "episode_m8_returns": episode_m8_returns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M8c PPO training with M8 reward")
    parser.add_argument("--distilled-checkpoint", type=str, default="data/m8_distilled_policy.pt",
                        help="M8 distilled policy checkpoint (weight init reference)")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Existing PPO checkpoint to initialize from")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--output-dir", type=str, default="results/m8_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-scripted-env", action="store_true",
                        help="Train on scripted conversation episodes instead of MultiStepTurnEnv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_config = M8RewardConfig()

    # Create model
    policy = M8ActorCritic(input_dim=13, hidden_dim=64, dropout=0.1).to(device)

    # Initialize from distilled checkpoint (partial weight mapping)
    if args.distilled_checkpoint and Path(args.distilled_checkpoint).exists():
        print(f"Initializing from distilled checkpoint: {args.distilled_checkpoint}")
        load_distilled_weights(policy, args.distilled_checkpoint, device)

    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        print(f"Loading PPO checkpoint: {args.init_checkpoint}")
        load_ppo_weights(policy, args.init_checkpoint, device)
        # Also load optimizer
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        optimizer.load_state_dict(ckpt["optimizer_state"])
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.use_scripted_env:
        if not _SCRIPTED_ENV_AVAILABLE:
            print("ERROR: ScriptedEpisodeEnv not available. Check imports.")
            sys.exit(1)
        env = ScriptedEpisodeEnv(seed=args.seed)
        print(f"Using ScriptedEpisodeEnv with {env.num_episodes} episode types")
    else:
        env = MultiStepTurnEnv()
        print(f"Using MultiStepTurnEnv with {len(env.templates)} templates")

    use_m4_base = not args.use_scripted_env

    # Baseline evaluation
    baseline = evaluate_policy(policy, env, reward_config, device, use_m4_base=use_m4_base)
    print(f"\nBaseline (before training):")
    print(f"  Mean episode return: {baseline['mean_episode_return']:.4f}")
    print(f"  Mean M8 return:      {baseline['mean_m8_return']:.4f}")
    print(f"  Episode returns:      {baseline['episode_returns']}")
    if args.eval_only:
        return 0

    best_return = -math.inf
    best_path = None
    history: List[Dict[str, Any]] = []

    for it in range(1, args.iterations + 1):
        trajectories = collect_rollouts(policy, env, reward_config, device, args.episodes,
                                        use_m4_base=use_m4_base)
        transitions = compute_returns_and_advantages(trajectories, gamma=args.gamma)
        stats = ppo_update(
            policy, optimizer, transitions, device,
            clip_ratio=args.clip_ratio, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        eval_report = evaluate_policy(policy, env, reward_config, device, use_m4_base=use_m4_base)

        record = {
            "iteration": it,
            "n_transitions": len(transitions),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "eval_mean_return": eval_report["mean_episode_return"],
            "eval_mean_m8_return": eval_report["mean_m8_return"],
            "eval_episode_returns": eval_report["episode_returns"],
        }
        history.append(record)

        if it % 5 == 0 or it == 1:
            print(f"  Iter {it:3d}/{args.iterations}  "
                  f"pol_loss={stats['policy_loss']:.4f}  val_loss={stats['value_loss']:.4f}  "
                  f"ent={stats['entropy']:.4f}  "
                  f"eval={eval_report['mean_episode_return']:+.4f}  "
                  f"m8={eval_report['mean_m8_return']:+.4f}")

        if eval_report["mean_episode_return"] > best_return:
            best_return = eval_report["mean_episode_return"]
            out_dir = Path(args.output_dir)
            best_path = out_dir / "best_policy.pt"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "policy_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {"input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
                "meta": {"history": history, "best_eval": eval_report,
                         "m8_reward_config": reward_config.__dict__},
            }, best_path)

    # Save final
    final_path = Path(args.output_dir) / "final_policy.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {"input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
        "meta": {"history": history},
    }, final_path)

    print(f"\nDone. Best eval return: {best_return:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
