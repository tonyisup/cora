#!/usr/bin/env python3
"""M9 Contrastive Substrate Training — 5-vowel wide retrain."""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
from m9_sound.substrate import AcousticSubstrate

SR = 16000
FRAME = 480
HIDDEN_DIM = 32

# Wider 5-vowel formants
VOWEL_FORMANTS = {
    'i': {'f1': 250, 'f2': 2600, 'f3': 3100, 'a1': 1.0, 'a2': 0.50, 'a3': 0.25},
    'E': {'f1': 500, 'f2': 2100, 'f3': 2600, 'a1': 1.0, 'a2': 0.55, 'a3': 0.30},
    'a': {'f1': 800, 'f2': 850,  'f3': 2350, 'a1': 1.0, 'a2': 0.75, 'a3': 0.40},
    'O': {'f1': 600, 'f2': 650,  'f3': 2300, 'a1': 1.0, 'a2': 0.65, 'a3': 0.35},
    'u': {'f1': 320, 'f2': 700,  'f3': 2150, 'a1': 1.0, 'a2': 0.45, 'a3': 0.20},
}
VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']


def gen_vowel(vowel, duration=0.5, f0=120.0, jitter=0.05, noise_db=-40.0, rng=None):
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


def _mel_features(frame, n_fft=512, n_mels=26, sr=SR):
    """Compute log-mel features for a single frame."""
    from m9_sound.vowel_transitions import MEL_FILTERBANK
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = np.asarray(frame, dtype=np.float32)[:n]
    spectrum = np.abs(np.fft.rfft(x))
    mel_spec = MEL_FILTERBANK @ spectrum
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = float(np.sqrt(np.mean(mel_spec ** 2)))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec


def forward_substrate(W_h, W_x, b, audio, hp_sos=None):
    if hp_sos is not None:
        audio = sosfilt(hp_sos, audio)
    n_frames = (len(audio) - FRAME) // FRAME + 1 if len(audio) >= FRAME else 0
    if n_frames == 0:
        return np.zeros(HIDDEN_DIM, dtype=np.float32), []
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    h_list = []
    pre_tanh_list = []
    x_norm_list = []
    for i in range(n_frames):
        frame = audio[i * FRAME:(i + 1) * FRAME].flatten().astype(np.float32)
        mel = _mel_features(frame)
        pre = W_h @ h + W_x @ mel + b
        h = np.tanh(pre)
        h_list.append(h.copy())
        pre_tanh_list.append(pre.copy())
        x_norm_list.append(mel.copy())
    lo = n_frames // 3
    hi = 2 * n_frames // 3
    if lo >= hi:
        lo, hi = 0, n_frames
    avg_h = np.mean(h_list[lo:hi], axis=0)
    cache = {
        'h_list': h_list, 'pre_tanh_list': pre_tanh_list,
        'x_norm_list': x_norm_list, 'lo': lo, 'hi': hi, 'n_frames': n_frames,
    }
    return avg_h.astype(np.float32), cache


def backward_substrate(d_avg_h, cache, W_h):
    h_list = cache['h_list']
    pre_list = cache['pre_tanh_list']
    x_list = cache['x_norm_list']
    lo = cache['lo']
    hi = cache['hi']
    n = cache['n_frames']
    n_avg = hi - lo
    d_W_h = np.zeros_like(W_h)
    d_W_x = np.zeros((HIDDEN_DIM, 26), dtype=np.float32)
    d_b = np.zeros(HIDDEN_DIM, dtype=np.float32)
    d_h_next = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in reversed(range(n)):
        if lo <= t < hi:
            d_h = d_avg_h / n_avg + d_h_next
        else:
            d_h = d_h_next
        d_pre = d_h * (1 - np.tanh(pre_list[t]) ** 2)
        d_b += d_pre
        if t > 0:
            d_W_h += np.outer(d_pre, h_list[t - 1])
        d_W_x += np.outer(d_pre, x_list[t])
        d_h_next = W_h.T @ d_pre
    return d_W_h.astype(np.float32), d_W_x.astype(np.float32), d_b.astype(np.float32)


def compute_triplet_loss_and_grad(embeddings, labels, margin=2.0):
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
            h_pos = centroids[v_a]
            h_neg = centroids[v_neg]
            d_pos = np.sum((h_a - h_pos) ** 2)
            d_neg = np.sum((h_a - h_neg) ** 2)
            viol = d_pos - d_neg + margin
            if viol > 0:
                loss += viol
                d_embed[i] += 2 * (h_a - h_pos) - 2 * (h_a - h_neg)
                d_pos_contrib = -2 * (h_a - h_pos) / counts[v_a]
                for j in range(n):
                    if labels[j] == v_a:
                        d_embed[j] += d_pos_contrib
                d_neg_contrib = 2 * (h_a - h_neg) / counts[v_neg]
                for j in range(n):
                    if labels[j] == v_neg:
                        d_embed[j] += d_neg_contrib
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
    loss = weight * np.mean(embeddings ** 4)
    n = embeddings.size
    d_embed = weight * 4 * embeddings ** 3 / n
    return loss, d_embed


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


def train(n_epochs=500, lr=0.001, n_per_vowel=300, margin=3.0,
          desat_weight=0.1, l2_weight=1e-4, seed=42):
    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

    # Start from current best checkpoint
    ckpt_path = Path('data/m9_substrate_spectral.ckpt.json')
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32) * 0.3
        W_x = np.array(ckpt['W_x'], dtype=np.float32) * 0.3
        b = np.array(ckpt['b'], dtype=np.float32)
        print("Initialized from spectral checkpoint (weights scaled to 0.3)")
    else:
        rng = np.random.RandomState(seed)
        scale = 0.05
        W_h = (rng.randn(HIDDEN_DIM, HIDDEN_DIM) * scale).astype(np.float32)
        W_x = (rng.randn(HIDDEN_DIM, FRAME) * scale).astype(np.float32)
        b = np.zeros(HIDDEN_DIM, dtype=np.float32)
        print("Initialized from random")

    print(f"Generating {n_per_vowel} samples per vowel ({n_per_vowel * len(VOWEL_NAMES)} total)...")
    dataset = generate_dataset(n_per_vowel=n_per_vowel, seed=seed)

    n_params = W_h.size + W_x.size + b.size
    print(f"Parameters: {n_params}")
    print(f"Epochs: {n_epochs}, lr: {lr}, margin: {margin}")
    print(f"Vowels: {VOWEL_NAMES}")
    print()

    m_Wh = np.zeros_like(W_h); v_Wh = np.zeros_like(W_h)
    m_Wx = np.zeros_like(W_x); v_Wx = np.zeros_like(W_x)
    m_b  = np.zeros_like(b);    v_b  = np.zeros_like(b)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    best_loss = float('inf')
    best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

    for epoch in range(n_epochs):
        embeddings = []
        labels = []
        caches = []
        for vowel, audio in dataset:
            h, cache = forward_substrate(W_h, W_x, b, audio, hp_sos)
            embeddings.append(h)
            labels.append(vowel)
            caches.append(cache)
        embeddings = np.array(embeddings, dtype=np.float32)

        tri_loss, d_tri, centroids = compute_triplet_loss_and_grad(
            embeddings, labels, margin=margin)
        desat_loss, d_desat = compute_desat_loss_and_grad(
            embeddings, weight=desat_weight)
        l2_loss = l2_weight * (np.sum(W_h ** 2) + np.sum(W_x ** 2) + np.sum(b ** 2))

        total_loss = tri_loss + desat_loss + l2_loss

        d_embeddings = d_tri + d_desat

        d_W_h = np.zeros_like(W_h)
        d_W_x = np.zeros_like(W_x)
        d_b = np.zeros_like(b)

        for i in range(len(dataset)):
            d_h = d_embeddings[i]
            dwh_i, dwx_i, db_i = backward_substrate(d_h, caches[i], W_h)
            d_W_h += dwh_i
            d_W_x += dwx_i
            d_b += db_i

        d_W_h += 2 * l2_weight * W_h
        d_W_x += 2 * l2_weight * W_x
        d_b += 2 * l2_weight * b

        max_grad_norm = 1.0
        flat_grad = np.concatenate([d_W_h.ravel(), d_W_x.ravel(), d_b.ravel()])
        grad_norm = np.linalg.norm(flat_grad)
        if grad_norm > max_grad_norm:
            clip_factor = max_grad_norm / grad_norm
            d_W_h *= clip_factor
            d_W_x *= clip_factor
            d_b *= clip_factor

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

        if total_loss < best_loss:
            best_loss = total_loss
            best_W_h = W_h.copy()
            best_W_x = W_x.copy()
            best_b = b.copy()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
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
            wmax = max(within_stats[v]['max'] for v in VOWEL_NAMES)
            ratio = bm / wmax if wmax > 0 else float('inf')
            print(f"  E{epoch:4d} loss={total_loss:.4f} (tri={tri_loss:.4f}) "
                  f"between_min={bm:.2f} within_max={wmax:.2f} ratio={ratio:.2f}")

    print(f"\nBest loss: {best_loss:.4f}")
    return best_W_h, best_W_x, best_b, dataset


def build_lookup(W_h, W_x, b, dataset, hp_sos, n_augment=200):
    from scipy.spatial import cKDTree
    embeddings = {v: [] for v in VOWEL_NAMES}
    for vowel, audio in dataset:
        h, _ = forward_substrate(W_h, W_x, b, audio, hp_sos)
        embeddings[vowel].append(h)
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
    all_h, all_labels = [], []
    for v in VOWEL_NAMES:
        all_h.extend(embeddings[v])
        all_labels.extend([v] * len(embeddings[v]))
    centroid_arr = np.array([centroids[v] for v in VOWEL_NAMES], dtype=np.float32)
    tree = cKDTree(centroid_arr)
    _, idxs = tree.query(np.array(all_h, dtype=np.float32), k=1)
    correct = sum(1 for i, idx in enumerate(idxs) if VOWEL_NAMES[idx] == all_labels[i])
    accuracy = correct / len(all_labels)
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


def main():
    parser = argparse.ArgumentParser(description='M9 5-vowel wide retrain')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-per-vowel', type=int, default=300)
    parser.add_argument('--margin', type=float, default=3.0)
    args = parser.parse_args()

    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

    W_h, W_x, b, dataset = train(
        n_epochs=args.epochs, lr=args.lr, n_per_vowel=args.n_per_vowel,
        margin=args.margin,
    )

    save_checkpoint(W_h, W_x, b, 'data/m9_substrate_spectral_v2.ckpt.json')
    centroids, within_stats, accuracy = build_lookup(W_h, W_x, b, dataset, hp_sos)
    save_lookup(centroids, within_stats, accuracy, 'data/m9_formant_lookup_v6.json')

    print("\nDone. Update harness to use v2 substrate + v6 lookup.")


if __name__ == '__main__':
    main()
