#!/usr/bin/env python3
"""M2 — Training Loop for Distilled Speaking Policy.

Trains a small MLP to mimic the deterministic policy (behavioral cloning)
and then distills from the teacher LLM (KL-divergence fine-tuning).

Usage:
  .venv/bin/python scripts/m2_train.py --mode bc --epochs 20
  .venv/bin/python scripts/m2_train.py --mode distill --epochs 10 --lr 1e-4
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


# ── Constants ────────────────────────────────────────────────────────────────

STATE_DIM = 5    # idle, listen, speak, repair, clarify
ATTR_DIM = 3     # silence, user/world, both
NUM_ACTIONS = 5  # speak, wait, clarify, repair, yield

STATE_INDEX = {"idle": 0, "listen": 1, "speak": 2, "repair": 3, "clarify": 4}
ATTR_INDEX = {"silence": 0, "self": 1, "world": 1, "both": 2}
ACTION_INDEX = {"speak": 0, "wait": 1, "clarify": 2, "repair": 3, "yield": 4}


# ── Feature Encoding ─────────────────────────────────────────────────────────

def encode_features(features: Dict[str, Any]) -> np.ndarray:
    """Encode a feature dict into a flat numpy array."""
    # State one-hot
    state_idx = features.get("state_idx", 0)
    state_onehot = np.zeros(STATE_DIM, dtype=np.float32)
    state_onehot[state_idx % STATE_DIM] = 1.0

    # Attribution one-hot
    attr_idx = features.get("attr_idx", 0)
    attr_onehot = np.zeros(ATTR_DIM, dtype=np.float32)
    attr_onehot[attr_idx % ATTR_DIM] = 1.0

    # Numeric features
    numeric = np.array([
        features.get("silence_duration_log", 0.0),
        features.get("vad_energy_db", -60.0) / 60.0,  # normalize to ~[-1, 0]
        float(features.get("has_queued_utterance", False)),
        float(features.get("user_speaking", False)),
        float(features.get("agent_speaking", False)),
    ], dtype=np.float32)

    return np.concatenate([state_onehot, attr_onehot, numeric])


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpeakingPolicyDataset(Dataset):
    """Loads JSONL training data for the speaking policy."""

    def __init__(self, paths: List[str], has_teacher: bool = False):
        self.samples: List[Dict[str, Any]] = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                print(f"WARNING: {p} not found, skipping")
                continue
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.samples.append(record)
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(self.samples)} samples from {len(paths)} file(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.samples[idx]
        features = record["features"]

        # Primary label: deterministic expert (hard label)
        det_idx = record.get("_det_action_idx", -1)
        if det_idx < 0 or det_idx >= NUM_ACTIONS:
            det_name = record.get("deterministic_action", record.get("action", "wait"))
            det_idx = ACTION_INDEX.get(det_name, 1)
        det_idx = max(0, min(det_idx, NUM_ACTIONS - 1))

        # Teacher label (soft label for distillation)
        teacher_idx = record.get("_teacher_action_idx", -1)
        if teacher_idx < 0 or teacher_idx >= NUM_ACTIONS:
            teacher_name = record.get("action", "wait")
            teacher_idx = ACTION_INDEX.get(teacher_name, 1)
        teacher_idx = max(0, min(teacher_idx, NUM_ACTIONS - 1))

        x = encode_features(features)
        result = {
            "features": torch.tensor(x, dtype=torch.float32),
            "action": torch.tensor(det_idx, dtype=torch.long),
            "teacher_action": torch.tensor(teacher_idx, dtype=torch.long),
            "has_teacher": torch.tensor(1.0 if record.get("deterministic_action") != record.get("action") else 0.0),
        }

        return result


# ── Model ────────────────────────────────────────────────────────────────────

class DistilledSpeakingPolicy(nn.Module):
    """Small MLP for speaking policy distillation."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        input_dim = STATE_DIM + ATTR_DIM + 5  # 5 numeric features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, features: Dict[str, Any]) -> Tuple[str, int]:
        """Predict action for a single feature dict. Returns (action_name, action_idx)."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(encode_features(features), dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            action_idx = probs.argmax(dim=-1).item()
        action_names = ["speak", "wait", "clarify", "repair", "yield"]
        return action_names[action_idx], action_idx


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, alpha: float = 0.8, temperature: float = 2.0) -> Dict[str, float]:
    """Train for one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        actions = batch["action"].to(device)

        logits = model(features)

        # Cross-entropy on hard labels
        ce_loss = F.cross_entropy(logits, actions)

        # KL divergence from teacher (only on disagreement cases)
        kl_loss = torch.tensor(0.0, device=device)
        if "teacher_action" in batch and "has_teacher" in batch:
            teacher_actions = batch["teacher_action"].to(device)
            has_teacher = batch["has_teacher"].to(device)  # 1.0 if disagreement
            if has_teacher.sum() > 0:
                # Only compute KL on disagreement cases
                disagree_mask = has_teacher > 0.5
                kl_loss = F.cross_entropy(
                    logits[disagree_mask], teacher_actions[disagree_mask], reduction="mean"
                )

        loss = alpha * ce_loss + (1 - alpha) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_ce += ce_loss.item() * features.size(0)
        total_kl += kl_loss.item() * features.size(0)
        correct += (logits.argmax(dim=-1) == actions).sum().item()
        total += features.size(0)

    return {
        "loss": total_loss / total,
        "ce_loss": total_ce / total,
        "kl_loss": total_kl / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    per_action_correct = {a: 0 for a in range(NUM_ACTIONS)}
    per_action_total = {a: 0 for a in range(NUM_ACTIONS)}

    for batch in loader:
        features = batch["features"].to(device)
        actions = batch["action"].to(device)
        logits = model(features)
        preds = logits.argmax(dim=-1)

        correct += (preds == actions).sum().item()
        total += features.size(0)

        for a in range(NUM_ACTIONS):
            mask = actions == a
            per_action_total[a] += mask.sum().item()
            per_action_correct[a] += ((preds == a) & mask).sum().item()

    action_names = ["speak", "wait", "clarify", "repair", "yield"]
    per_action_acc = {}
    for a in range(NUM_ACTIONS):
        if per_action_total[a] > 0:
            per_action_acc[action_names[a]] = per_action_correct[a] / per_action_total[a]

    return {
        "accuracy": correct / total,
        "per_action_accuracy": per_action_acc,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Train distilled speaking policy")
    parser.add_argument("--mode", required=True, choices=["bc", "distill"], help="bc=behavioral cloning, distill=teacher distillation")
    parser.add_argument("--data", type=str, nargs="+", default=["results/m2_data_deterministic.jsonl"])
    parser.add_argument("--teacher-data", type=str, nargs="+", default=None)
    parser.add_argument("--hardneg-data", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results/m2_model_checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for CE loss (1-alpha for KL)")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset
    all_paths = list(args.data)
    if args.hardneg_data:
        all_paths.extend(args.hardneg_data)

    dataset = SpeakingPolicyDataset(all_paths)

    # Compute class weights for weighted sampling
    action_counts = collections.Counter()
    for s in dataset.samples:
        a = s.get("action", "wait")
        action_counts[a] += 1
    print("Class distribution:")
    for a, c in sorted(action_counts.items()):
        print(f"  {a}: {c} ({c/len(dataset):.2%})")

    # Inverse-frequency class weights for loss
    total = sum(action_counts.values())
    class_weights = torch.tensor([
        total / max(action_counts.get(a, 1), 1)
        for a in ["speak", "wait", "clarify", "repair", "yield"]
    ], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * NUM_ACTIONS  # normalize to mean=1
    print(f"Class weights: {class_weights.tolist()}")

    # Oversample minority classes: weight each sample by inverse class freq
    sample_weights = []
    for s in dataset.samples:
        a = s.get("action", "wait")
        idx = ACTION_INDEX.get(a, 1)
        sample_weights.append(class_weights[idx].item())

    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    # Weighted random split using oversampling for train
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    # Use WeightedRandomSampler for training to balance classes
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = torch.utils.data.WeightedRandomSampler(
        train_weights, num_samples=len(train_indices), replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = DistilledSpeakingPolicy(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Mode: {args.mode}, Epochs: {args.epochs}, LR: {args.lr}, Alpha: {args.alpha}")

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, alpha=args.alpha, temperature=args.temperature)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={train_metrics['loss']:.4f} ce={train_metrics['ce_loss']:.4f} kl={train_metrics['kl_loss']:.4f} | "
              f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_accuracy": best_val_acc,
                "args": vars(args),
            }, ckpt_path)

    # Save final model and history
    final_path = output_dir / "final_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": args.epochs,
        "val_accuracy": val_metrics["accuracy"],
        "args": vars(args),
    }, final_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, default=str))

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")

    # Print per-action accuracy on val set
    final_val = evaluate(model, val_loader, device)
    print("\nPer-action validation accuracy:")
    for action, acc in sorted(final_val.get("per_action_accuracy", {}).items()):
        print(f"  {action:<10} {acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
