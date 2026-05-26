#!/usr/bin/env python3
"""M8b — Train distilled speaking policy via behavioral cloning.

Trains a small MLP to imitate the heuristic speaking policy using
collected (features, action) demonstration pairs.

Usage:
    .venv/bin/python scripts/m8_train_distilled.py \\
        --data /tmp/m8_demonstrations.jsonl \\
        --checkpoint data/m8_distilled_policy.pt \\
        --epochs 50

    .venv/bin/python scripts/m8_train_distilled.py \\
        --data /tmp/m8_demonstrations.jsonl \\
        --checkpoint data/m8_distilled_policy.pt \\
        --eval-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from m8_policy.model import M8DistilledPolicy, encode_m8_features, ACTION_NAMES


STATE_INDEX = {"idle": 0, "listening": 1, "speaking": 2, "repair": 3, "clarification": 4}


class M8Dataset(Dataset):
    """Dataset of (features, action) pairs for M8 policy training."""

    def __init__(self, paths: List[str]):
        self.samples: List[Tuple[torch.Tensor, int]] = []
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
                    except json.JSONDecodeError:
                        continue
                    features = record["features"]
                    action_name = record.get("action", "wait")
                    action_idx = ACTION_NAMES.index(action_name) if action_name in ACTION_NAMES else 1
                    x = encode_m8_features(features)
                    self.samples.append((x, action_idx))
        print(f"Loaded {len(self.samples)} samples from {len(paths)} file(s)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


def train_epoch(model: M8DistilledPolicy, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: M8DistilledPolicy, loader: DataLoader,
             device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    action_correct: Dict[str, int] = {a: 0 for a in ACTION_NAMES}
    action_total: Dict[str, int] = {a: 0 for a in ACTION_NAMES}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * len(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        n += len(x)

        for pred, true in zip(preds.cpu().tolist(), y.cpu().tolist()):
            action_total[ACTION_NAMES[true]] = action_total.get(ACTION_NAMES[true], 0) + 1
            if pred == true:
                action_correct[ACTION_NAMES[true]] = action_correct.get(ACTION_NAMES[true], 0) + 1

    avg_loss = total_loss / max(n, 1)
    accuracy = correct / max(n, 1)

    # Per-action accuracy
    per_action = {}
    for a in ACTION_NAMES:
        t = action_total.get(a, 0)
        c = action_correct.get(a, 0)
        per_action[a] = c / max(t, 1) if t > 0 else 0.0

    return avg_loss, accuracy, per_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M8 distilled policy")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to demonstrations JSONL file")
    parser.add_argument("--checkpoint", type=str, default="data/m8_distilled_policy.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = M8DistilledPolicy(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Load checkpoint if eval-only
    if args.eval_only:
        if not Path(args.checkpoint).exists():
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        # Load data
        dataset = M8Dataset([args.data])
        if len(dataset) == 0:
            print("ERROR: No training data loaded")
            sys.exit(1)

        # Split
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        print(f"Training: {train_size} samples, Validation: {val_size} samples")
        print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Device: {device}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc, per_action = evaluate(model, val_loader, device)
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best checkpoint
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "input_dim": 21,
                        "num_actions": 5,
                    },
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                }, args.checkpoint)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")

    # Final evaluation
    print("\nEvaluating distilled policy on all data...")
    full_loader = DataLoader(M8Dataset([args.data]), batch_size=args.batch_size)
    _, acc, per_action = evaluate(model, full_loader, device)
    print(f"  Overall accuracy: {acc:.4f}")
    print(f"  Per-action accuracy:")
    for a in ACTION_NAMES:
        print(f"    {a:10s}: {per_action.get(a, 0.0):.4f}")


if __name__ == "__main__":
    main()
