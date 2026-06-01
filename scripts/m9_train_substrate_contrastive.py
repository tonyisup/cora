#!/usr/bin/env python3
"""M9 Contrastive Substrate Training.

Trains the AcousticSubstrate RNN so that different vowels produce
well-separated hidden states, using triplet (max-margin) loss.

Key changes from the original substrate:
1. Contrastive loss — explicitly pushes different vowel embeddings apart
2. Desaturation penalty — discourages tanh saturation (dims pinned at ±1)
3. No per-frame RMS normalization in loss — preserves spectral balance cues
4. Heavy augmentation — formant jitter, pitch variation, noise floor

The substrate step() still does RMS norm internally (we can't change that
without modifying the class). Instead, we scale the audio before stepping
so the RMS norm sees the relative amplitude differences.

Training uses analytic backprop through the RNN + triplet loss.

Usage:
    python scripts/m9_train_substrate_contrastive.py
    python scripts/m9_train_substrate_contrastive.py --eval-only
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.substrate import AcousticSubstrate

SR = 16000
FRAME = 480
HIDDEN_DIM = 32

VOWEL_FORMANTS = {
    'i': {'f1': 280, 'f2': 2250, 'f3': 2900, 'a1': 1.0, 'a2': 0.4, 'a3': 0.2},
    'a': {'f1': 730, 'f2': 1090, 'f3': 2440, 'a1': 1.0, 'a2': 0.7, 'a3': 0.35},
    'u': {'f1': 300, 'f2': 870,  'f3': 2240, 'a1': 1.0, 'a2': 0.4, 'a3': 0.15},
}
VOWEL_NAMES = list(VOWEL_FORMANTS.keys())


# ── Synthetic vowel generation ────────────────────────────────────────────

def gen_vowel(vowel, duration=0.5, f0=120.0, jitter=0.05, noise_db=-40.0,
              rng=None):
    if rng is None:
        rng = np.random.RandomState()
    t_def = VOWEL_FORMANTS[vowel]
    f1 = t_def['f1'] * (1 + jitter * rng.randn())
    f2 = t_def['f2'] * (1 + jitter * rng.randn())
    f3 = t_def['f3'] * (1 + jitter * rng.randn())
    a1 = t_def['a1']
    a2 = max(0.05, t_def['a2'] * (1 + jitter * rng.randn()))
    a3 = max(0.05, t_def['a3'] * (1 + jitter * rng.randn()))
    n = int(SR * duration)
    t = np.arange(n, dtype=np.float32) / SR
    phase0 = rng.uniform(0, 2 * np.pi)
    sig = (0.3 * np.sin(2 * np.pi * f0 * t + phase0)
           + a1 * np.sin(2 * np.pi * f1 * t + phase0 * 1.3)
           + a2 * np.sin(2 * np.pi * f2 * t + phase0 * 0.7)
           + a3 * np.sin(2 * np.pi * f3 * t + phase0 * 1.1))
    if noise_db > -60:
        sig += (10 ** (noise_db / 20)) * rng.randn(n).astype(np.float32)
    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = sig / peak
    return sig.astype(np.float32)


# ── Forward + backward through the substrate RNN ──────────────────────────

def forward_substrate(W_h, W_x, b, audio, hp_sos=None):
    """Run RNN forward, return average hidden state over middle third.

    Uses the same logic as AcousticSubstrate.step: RMS normalize each frame,
    then h = tanh(W_h @ h + W_x @ x + b).

    Returns: avg_h (HIDDEN_DIM,), and cache for backprop.
    """
    if hp_sos is not None:
        audio = sosfilt(hp_sos, audio)

    n_frames = (len(audio) - FRAME) // FRAME + 1 if len(audio) >= FRAME else 0
    if n_frames == 0:
        return np.zeros(HIDDEN_DIM, dtype=np.float32), []

    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    h_list = []
    pre_tanh_list = []  # pre-activation values for backprop
    x_norm_list = []    # normalized input frames

    for i in range(n_frames):
        frame = audio[i * FRAME:(i + 1) * FRAME].flatten().astype(np.float32)
        # No RMS normalization (preserves spectral balance cues)
        x_norm = frame
        pre = W_h @ h + W_x @ x_norm + b
        h = np.tanh(pre)
        h_list.append(h.copy())
        pre_tanh_list.append(pre.copy())
        x_norm_list.append(x_norm.copy())

    # Average over middle third
    lo = n_frames // 3
    hi = 2 * n_frames // 3
    if lo >= hi:
        lo, hi = 0, n_frames
    avg_h = np.mean(h_list[lo:hi], axis=0)

    cache = {
        'h_list': h_list,
        'pre_tanh_list': pre_tanh_list,
        'x_norm_list': x_norm_list,
        'lo': lo,
        'hi': hi,
        'n_frames': n_frames,
    }
    return avg_h.astype(np.float32), cache


def backward_substrate(d_avg_h, cache, W_h):
    """Backprop through the RNN to get gradients w.r.t. W_h, W_x, b.

    d_avg_h: gradient of loss w.r.t. avg_h (HIDDEN_DIM,)
    Returns: d_W_h, d_W_x, d_b
    """
    h_list = cache['h_list']
    pre_list = cache['pre_tanh_list']
    x_list = cache['x_norm_list']
    lo = cache['lo']
    hi = cache['hi']
    n = cache['n_frames']
    n_avg = hi - lo

    d_W_h = np.zeros_like(W_h)
    d_W_x = np.zeros((HIDDEN_DIM, FRAME), dtype=np.float32)
    d_b = np.zeros(HIDDEN_DIM, dtype=np.float32)

    # d_avg_h is the gradient w.r.t. the average over [lo, hi)
    # So each h_i for i in [lo, hi) gets d_avg_h / n_avg

    # Backprop through time
    d_h_next = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in reversed(range(n)):
        if lo <= t < hi:
            d_h = d_avg_h / n_avg + d_h_next
        else:
            d_h = d_h_next

        # tanh derivative
        d_pre = d_h * (1 - np.tanh(pre_list[t]) ** 2)

        # Accumulate parameter gradients
        d_b += d_pre
        if t > 0:
            d_W_h += np.outer(d_pre, h_list[t - 1])
        d_W_x += np.outer(d_pre, x_list[t])

        # Propagate to previous timestep
        d_h_next = W_h.T @ d_pre

    return d_W_h.astype(np.float32), d_W_x.astype(np.float32), d_b.astype(np.float32)


# ── Loss computation ──────────────────────────────────────────────────────

def compute_triplet_loss_and_grad(embeddings, labels, margin=2.0):
    """Compute triplet loss (anchor-positive vs anchor-negative)
    and gradient w.r.t. embeddings.

    Uses class centroids as positive/negative targets.
    """
    n = len(embeddings)
    centroids = {}
    counts = {}
    for v in VOWEL_NAMES:
        idx = [i for i, l in enumerate(labels) if l == v]
        centroids[v] = np.mean(embeddings[idx], axis=0)
        counts[v] = len(idx)

    loss = 0.0
    d_embed = np.zeros_like(embeddings)

    for i in range(n):
        v_a = labels[i]
        h_a = embeddings[i]

        for v_neg in VOWEL_NAMES:
            if v_neg == v_a:
                continue

            # Positive: centroid of own class
            h_pos = centroids[v_a]
            # Negative: centroid of other class
            h_neg = centroids[v_neg]

            d_pos = np.sum((h_a - h_pos) ** 2)
            d_neg = np.sum((h_a - h_neg) ** 2)
            viol = d_pos - d_neg + margin

            if viol > 0:
                loss += viol

                # Gradient w.r.t. h_a
                # d/dd_pos * dd_pos/dh_a = 2*(h_a - h_pos)
                # d/dd_neg * dd_neg/dh_a = -2*(h_a - h_neg)
                d_embed[i] += 2 * (h_a - h_pos) - 2 * (h_a - h_neg)

                # Gradient w.r.t. h_pos (centroid of anchor class)
                # Affects all samples in anchor class
                # d/dd_pos * dd_pos/dh_pos = -2*(h_a - h_pos) / counts[v_a]
                d_pos_contrib = -2 * (h_a - h_pos) / counts[v_a]
                for j in range(n):
                    if labels[j] == v_a:
                        d_embed[j] += d_pos_contrib

                # Gradient w.r.t. h_neg (centroid of negative class)
                # d/dd_neg * dd_neg/dh_neg = 2*(h_a - h_neg) / counts[v_neg]
                d_neg_contrib = 2 * (h_a - h_neg) / counts[v_neg]
                for j in range(n):
                    if labels[j] == v_neg:
                        d_embed[j] += d_neg_contrib

    # Count active triplets for normalization
    n_active = 0
    for i in range(n):
        v_a = labels[i]
        for v_neg in VOWEL_NAMES:
            if v_neg == v_a:
                continue
            h_pos = centroids[v_a]
            h_neg = centroids[v_neg]
            d_pos = np.sum((embeddings[i] - h_pos) ** 2)
            d_neg = np.sum((embeddings[i] - h_neg) ** 2)
            if d_pos - d_neg + margin > 0:
                n_active += 1

    if n_active > 0:
        loss /= n_active
        d_embed /= n_active

    return loss, d_embed, centroids


def compute_desat_loss_and_grad(embeddings, weight=0.1):
    """Desaturation penalty: penalize |h|^4 (grows fast near ±1)."""
    loss = weight * np.mean(embeddings ** 4)
    # d/dh of (h^4) = 4*h^3, averaged over all elements
    n = embeddings.size
    d_embed = weight * 4 * embeddings ** 3 / n
    return loss, d_embed


# ── Dataset generation ────────────────────────────────────────────────────

def generate_dataset(n_per_vowel=300, seed=42):
    rng = np.random.RandomState(seed)
    dataset = []
    for vowel in VOWEL_NAMES:
        for _ in range(n_per_vowel):
            f0 = rng.uniform(100, 180)
            jitter = rng.uniform(0.02, 0.08)
            noise_db = rng.uniform(-45, -30)
            dur = rng.uniform(0.4, 0.8)
            audio = gen_vowel(vowel, duration=dur, f0=f0, jitter=jitter,
                              noise_db=noise_db, rng=rng)
            dataset.append((vowel, audio))
    return dataset


# ── Training ──────────────────────────────────────────────────────────────

def train(n_epochs=300, lr=0.001, n_per_vowel=300, margin=2.0,
          desat_weight=0.1, l2_weight=1e-4, seed=42, fresh=False):
    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

    # Initialize: prefer v2 checkpoint if it exists AND is compatible
    # If substrate.step was changed (e.g. RMS norm removed), force reinit
    v2_path = Path('data/m9_substrate_formant_v2.ckpt.json')
    v1_path = Path('data/m9_substrate_formant.ckpt.json')
    
    # Check for fresh flag or missing v2
    if v2_path.exists() and not fresh:
        ckpt = json.load(open(v2_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32)
        W_x = np.array(ckpt['W_x'], dtype=np.float32)
        b = np.array(ckpt['b'], dtype=np.float32)
        print("Resumed from v2 checkpoint")
    elif v1_path.exists():
        ckpt = json.load(open(v1_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32) * 0.3
        W_x = np.array(ckpt['W_x'], dtype=np.float32) * 0.3
        b = np.array(ckpt['b'], dtype=np.float32)
        print("Initialized from v1 checkpoint (weights scaled to 0.3)")
    else:
        rng = np.random.RandomState(seed)
        scale = 0.05
        W_h = (rng.randn(HIDDEN_DIM, HIDDEN_DIM) * scale).astype(np.float32)
        W_x = (rng.randn(HIDDEN_DIM, FRAME) * scale).astype(np.float32)
        b = np.zeros(HIDDEN_DIM, dtype=np.float32)
        print("Initialized from random")

    print(f"Generating {n_per_vowel} samples per vowel ({n_per_vowel * 3} total)...")
    dataset = generate_dataset(n_per_vowel=n_per_vowel, seed=seed)

    n_params = W_h.size + W_x.size + b.size
    print(f"Parameters: {n_params} (W_h={W_h.size}, W_x={W_x.size}, b={b.size})")
    print(f"Epochs: {n_epochs}, lr: {lr}, margin: {margin}")
    print(f"Desaturation weight: {desat_weight}, L2 weight: {l2_weight}")
    print()

    # Adam optimizer state
    m_Wh = np.zeros_like(W_h); v_Wh = np.zeros_like(W_h)
    m_Wx = np.zeros_like(W_x); v_Wx = np.zeros_like(W_x)
    m_b  = np.zeros_like(b);    v_b  = np.zeros_like(b)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')
    best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

    for epoch in range(n_epochs):
        # ── Forward pass: encode all samples ──
        embeddings = []
        labels = []
        caches = []
        for vowel, audio in dataset:
            h, cache = forward_substrate(W_h, W_x, b, audio, hp_sos)
            embeddings.append(h)
            labels.append(vowel)
            caches.append(cache)
        embeddings = np.array(embeddings, dtype=np.float32)

        # ── Compute losses ──
        tri_loss, d_tri, centroids = compute_triplet_loss_and_grad(
            embeddings, labels, margin=margin)
        desat_loss, d_desat = compute_desat_loss_and_grad(
            embeddings, weight=desat_weight)
        l2_loss = l2_weight * (np.sum(W_h ** 2) + np.sum(W_x ** 2) + np.sum(b ** 2))

        total_loss = tri_loss + desat_loss + l2_loss

        # ── Backprop through substrate ──
        d_embeddings = d_tri + d_desat  # combined gradient w.r.t. embeddings

        d_W_h = np.zeros_like(W_h)
        d_W_x = np.zeros_like(W_x)
        d_b = np.zeros_like(b)

        for i in range(len(dataset)):
            d_h = d_embeddings[i]
            dwh_i, dwx_i, db_i = backward_substrate(d_h, caches[i], W_h)
            d_W_h += dwh_i
            d_W_x += dwx_i
            d_b += db_i

        # L2 gradient
        d_W_h += 2 * l2_weight * W_h
        d_W_x += 2 * l2_weight * W_x
        d_b += 2 * l2_weight * b

        # ── Adam update ──
        t_step = epoch + 1
        for param_name in ['W_h', 'W_x', 'b']:
            g = d_W_h if param_name == 'W_h' else (d_W_x if param_name == 'W_x' else d_b)
            m = m_Wh if param_name == 'W_h' else (m_Wx if param_name == 'W_x' else m_b)
            v = v_Wh if param_name == 'W_h' else (v_Wx if param_name == 'W_x' else v_b)

            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)

            if param_name == 'W_h':
                W_h -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
            elif param_name == 'W_x':
                W_x -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
            else:
                b -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        # ── Track best ──
        if total_loss < best_loss:
            best_loss = total_loss
            best_W_h = W_h.copy()
            best_W_x = W_x.copy()
            best_b = b.copy()

        # ── Logging ──
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            # Compute metrics
            between_dists = []
            for i, vi in enumerate(VOWEL_NAMES):
                for j, vj in enumerate(VOWEL_NAMES):
                    if j <= i:
                        continue
                    between_dists.append(np.linalg.norm(centroids[vi] - centroids[vj]))

            within_stats = {}
            for v in VOWEL_NAMES:
                idx = [i for i, l in enumerate(labels) if l == v]
                dists_v = [np.linalg.norm(embeddings[i] - centroids[v]) for i in idx]
                within_stats[v] = {'mean': float(np.mean(dists_v)),
                                    'max': float(np.max(dists_v))}

            bm = min(between_dists)
            wm = np.mean([within_stats[v]['mean'] for v in VOWEL_NAMES])
            wmax = max(within_stats[v]['max'] for v in VOWEL_NAMES)
            ratio = bm / wmax if wmax > 0 else float('inf')

            print(f"  E{epoch:3d} loss={total_loss:.4f} (tri={tri_loss:.4f} desat={desat_loss:.4f} l2={l2_loss:.6f}) "
                  f"between_min={bm:.2f} within_mean={wm:.2f} within_max={wmax:.2f} ratio={ratio:.2f}")

    print(f"\nBest loss: {best_loss:.4f}")
    return best_W_h, best_W_x, best_b, dataset


# ── Lookup table ──────────────────────────────────────────────────────────

def build_lookup(W_h, W_x, b, dataset, hp_sos, n_augment=200):
    from scipy.spatial import cKDTree

    # Encode dataset
    embeddings = {v: [] for v in VOWEL_NAMES}
    for vowel, audio in dataset:
        h, _ = forward_substrate(W_h, W_x, b, audio, hp_sos)
        embeddings[vowel].append(h)

    # Extra augmented samples for robust centroids
    rng = np.random.RandomState(123)
    for vowel in VOWEL_NAMES:
        for _ in range(n_augment):
            f0 = rng.uniform(100, 180)
            audio = gen_vowel(vowel, duration=0.5, f0=f0,
                              jitter=rng.uniform(0.02, 0.08),
                              noise_db=rng.uniform(-45, -30), rng=rng)
            h, _ = forward_substrate(W_h, W_x, b, audio, hp_sos)
            embeddings[vowel].append(h)

    centroids = {v: np.mean(embeddings[v], axis=0).astype(np.float32)
                 for v in VOWEL_NAMES}

    # Accuracy
    all_h, all_labels = [], []
    for v in VOWEL_NAMES:
        all_h.extend(embeddings[v])
        all_labels.extend([v] * len(embeddings[v]))
    centroid_arr = np.array([centroids[v] for v in VOWEL_NAMES], dtype=np.float32)
    tree = cKDTree(centroid_arr)
    _, idxs = tree.query(np.array(all_h, dtype=np.float32), k=1)
    correct = sum(1 for i, idx in enumerate(idxs) if VOWEL_NAMES[idx] == all_labels[i])
    accuracy = correct / len(all_labels)

    # Stats
    between_dists = []
    for i, vi in enumerate(VOWEL_NAMES):
        for j, vj in enumerate(VOWEL_NAMES):
            if j <= i:
                continue
            between_dists.append(np.linalg.norm(centroids[vi] - centroids[vj]))

    within_stats = {}
    for v in VOWEL_NAMES:
        h_arr = np.array(embeddings[v], dtype=np.float32)
        dists_v = np.linalg.norm(h_arr - centroids[v], axis=1)
        within_stats[v] = {'mean': float(np.mean(dists_v)),
                            'std': float(np.std(dists_v)),
                            'max': float(np.max(dists_v))}

    print("\n=== Lookup Table ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(all_labels)})")
    print(f"Between: {', '.join(f'{d:.2f}' for d in between_dists)} (min={min(between_dists):.2f})")
    for v in VOWEL_NAMES:
        s = within_stats[v]
        print(f"  Within {v}: mean={s['mean']:.2f} std={s['std']:.2f} max={s['max']:.2f}")
    wmax = max(s['max'] for s in within_stats.values())
    print(f"Separation ratio: {min(between_dists)/wmax:.2f}")

    return centroids, within_stats, accuracy


def save_checkpoint(W_h, W_x, b, path):
    data = {'W_h': W_h.tolist(), 'W_x': W_x.tolist(), 'b': b.tolist(),
            'hidden_dim': HIDDEN_DIM, 'frame_size': FRAME, 'sr': SR}
    Path(path).write_text(json.dumps(data))
    print(f"Saved: {path}")


def save_lookup(centroids, within_stats, accuracy, path):
    lookup = {}
    for v in VOWEL_NAMES:
        lookup[v] = {'centroid': centroids[v].tolist(), 'template': VOWEL_FORMANTS[v]}
    data = {'lookup': lookup, 'vowel_names': VOWEL_NAMES,
            'accuracy': float(accuracy),
            'within_stats': {v: within_stats[v] for v in VOWEL_NAMES}}
    Path(path).write_text(json.dumps(data))
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='M9 Contrastive Substrate Training')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-per-vowel', type=int, default=300)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--desat-weight', type=float, default=0.1)
    parser.add_argument('--l2-weight', type=float, default=1e-4)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--fresh', action='store_true',
                        help='Force fresh init (ignore v2 checkpoint)')
    args = parser.parse_args()

    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

    if args.eval_only:
        ckpt_path = Path('data/m9_substrate_formant_v2.ckpt.json')
        if not ckpt_path.exists():
            ckpt_path = Path('data/m9_substrate_formant.ckpt.json')
        ckpt = json.load(open(ckpt_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32)
        W_x = np.array(ckpt['W_x'], dtype=np.float32)
        b = np.array(ckpt['b'], dtype=np.float32)
        dataset = generate_dataset(n_per_vowel=args.n_per_vowel, seed=42)
        centroids, within_stats, accuracy = build_lookup(W_h, W_x, b, dataset, hp_sos)
        save_lookup(centroids, within_stats, accuracy, 'data/m9_formant_lookup_v4.json')
        return

    W_h, W_x, b, dataset = train(
        n_epochs=args.epochs, lr=args.lr, n_per_vowel=args.n_per_vowel,
        margin=args.margin, desat_weight=args.desat_weight,
        l2_weight=args.l2_weight, fresh=args.fresh,
    )

    save_checkpoint(W_h, W_x, b, 'data/m9_substrate_formant_v2.ckpt.json')
    centroids, within_stats, accuracy = build_lookup(W_h, W_x, b, dataset, hp_sos)
    save_lookup(centroids, within_stats, accuracy, 'data/m9_formant_lookup_v4.json')

    print("\nDone. Update m9_live_vowel.py to use v2 substrate + v4 lookup, then live test.")


if __name__ == '__main__':
    main()
