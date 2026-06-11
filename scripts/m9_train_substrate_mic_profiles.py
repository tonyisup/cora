#!/usr/bin/env python3
"""M9 Substrate Training — aggressive speaker-mic augmentation."""

import sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt, cheby1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))

SR = 16000
FRAME = 480
HIDDEN_DIM = 32

VOWEL_FORMANTS = {
    'i': {'f1': 230, 'f2': 2800, 'f3': 3300, 'a1': 1.0, 'a2': 0.55, 'a3': 0.30},
    'E': {'f1': 480, 'f2': 2200, 'f3': 2700, 'a1': 1.0, 'a2': 0.60, 'a3': 0.35},
    'a': {'f1': 900, 'f2': 700,  'f3': 2300, 'a1': 1.0, 'a2': 0.80, 'a3': 0.45},
    'O': {'f1': 620, 'f2': 550,  'f3': 2250, 'a1': 1.0, 'a2': 0.70, 'a3': 0.40},
    'u': {'f1': 340, 'f2': 600,  'f3': 2100, 'a1': 1.0, 'a2': 0.50, 'a3': 0.25},
}
VOWEL_NAMES = ['i', 'E', 'a', 'O', 'u']

# Multiple speaker-mic profiles for augmentation
HP_SOS = butter(4, 100, btype='high', fs=SR, output='sos')
LP_SOS_4K = cheby1(4, 1, 4000, btype='low', fs=SR, output='sos')
LP_SOS_3K = cheby1(4, 1, 3000, btype='low', fs=SR, output='sos')
LP_SOS_2K = cheby1(4, 1, 2000, btype='low', fs=SR, output='sos')

def apply_mic(audio, lp_sos):
    audio = sosfilt(HP_SOS, audio).astype(np.float32)
    audio = sosfilt(lp_sos, audio).astype(np.float32)
    return audio

def gen_vowel(vowel, duration=0.5, f0=120.0, jitter=0.05, noise_db=-40.0,
              lp_sos=None, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    t = VOWEL_FORMANTS[vowel]
    f1 = t['f1'] * (1 + jitter * rng.randn())
    f2 = t['f2'] * (1 + jitter * rng.randn())
    f3 = t['f3'] * (1 + jitter * rng.randn())
    a1 = t['a1']
    a2 = max(0.05, t['a2'] * (1 + jitter * rng.randn()))
    a3 = max(0.05, t['a3'] * (1 + jitter * rng.randn()))
    n = int(SR * duration)
    t_arr = np.arange(n, dtype=np.float32) / SR
    phase0 = rng.uniform(0, 2 * np.pi)
    sig = (0.3 * np.sin(2 * np.pi * f0 * t_arr + phase0)
           + a1 * np.sin(2 * np.pi * f1 * t_arr + phase0 * 1.3)
           + a2 * np.sin(2 * np.pi * f2 * t_arr + phase0 * 0.7)
           + a3 * np.sin(2 * np.pi * f3 * t_arr + phase0 * 1.1))
    if noise_db > -60:
        sig += (10 ** (noise_db / 20)) * rng.randn(n).astype(np.float32)
    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig = sig / peak
    sig = sig.astype(np.float32)
    if lp_sos is not None:
        sig = apply_mic(sig, lp_sos)
    return sig

def _mel_features(frame, n_fft=512, n_mels=26):
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
    h_list, pre_tanh_list, x_norm_list = [], [], []
    for i in range(n_frames):
        frame = audio[i * FRAME:(i + 1) * FRAME].flatten().astype(np.float32)
        mel = _mel_features(frame)
        pre = W_h @ h + W_x @ mel + b
        h = np.tanh(pre)
        h_list.append(h.copy())
        pre_tanh_list.append(pre.copy())
        x_norm_list.append(mel.copy())
    lo, hi = n_frames // 3, 2 * n_frames // 3
    if lo >= hi:
        lo, hi = 0, n_frames
    avg_h = np.mean(h_list[lo:hi], axis=0)
    cache = {'h_list': h_list, 'pre_tanh_list': pre_tanh_list,
             'x_norm_list': x_norm_list, 'lo': lo, 'hi': hi, 'n_frames': n_frames}
    return avg_h.astype(np.float32), cache

def backward_substrate(d_avg_h, cache, W_h):
    h_list, pre_list, x_list = cache['h_list'], cache['pre_tanh_list'], cache['x_norm_list']
    lo, hi, n = cache['lo'], cache['hi'], cache['n_frames']
    n_avg = hi - lo
    d_W_h, d_W_x, d_b = np.zeros_like(W_h), np.zeros((HIDDEN_DIM, 26), dtype=np.float32), np.zeros(HIDDEN_DIM, dtype=np.float32)
    d_h_next = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in reversed(range(n)):
        d_h = (d_avg_h / n_avg + d_h_next) if lo <= t < hi else d_h_next
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
            h_pos, h_neg = centroids[v_a], centroids[v_neg]
            d_pos = np.sum((h_a - h_pos) ** 2)
            d_neg = np.sum((h_a - h_neg) ** 2)
            viol = d_pos - d_neg + margin
            if viol > 0:
                loss += viol
                d_embed[i] += 2 * (h_a - h_pos) - 2 * (h_a - h_neg)
                d_pos_c = -2 * (h_a - h_pos) / counts[v_a]
                for j in range(n):
                    if labels[j] == v_a:
                        d_embed[j] += d_pos_c
                d_neg_c = 2 * (h_a - h_neg) / counts[v_neg]
                for j in range(n):
                    if labels[j] == v_neg:
                        d_embed[j] += d_neg_c
    n_active = 0
    for i in range(n):
        v_a = labels[i]
        for v_neg in VOWEL_NAMES:
            if v_neg == v_a:
                continue
            if np.sum((embeddings[i] - centroids[v_a]) ** 2) - np.sum((embeddings[i] - centroids[v_neg]) ** 2) + margin > 0:
                n_active += 1
    if n_active > 0:
        loss /= n_active
        d_embed /= n_active
    return loss, d_embed, centroids

def compute_desat_loss_and_grad(embeddings, weight=0.1):
    loss = weight * np.mean(embeddings ** 4)
    d_embed = weight * 4 * embeddings ** 3 / embeddings.size
    return loss, d_embed

def generate_dataset(n_per_vowel=300, seed=42):
    """Generate dataset with multiple speaker-mic profiles."""
    rng = np.random.RandomState(seed)
    dataset = []
    lp_profiles = [None, LP_SOS_4K, LP_SOS_3K, LP_SOS_2K]
    
    for vowel in VOWEL_NAMES:
        for _ in range(n_per_vowel):
            f0 = rng.uniform(100, 180)
            jitter = rng.uniform(0.02, 0.08)
            noise_db = rng.uniform(-45, -30)
            dur = rng.uniform(0.4, 0.8)
            # Randomly select a mic profile
            lp_sos = lp_profiles[rng.randint(len(lp_profiles))]
            audio = gen_vowel(vowel, duration=dur, f0=f0, jitter=jitter,
                              noise_db=noise_db, lp_sos=lp_sos, rng=rng)
            dataset.append((vowel, audio))
    return dataset

def train(n_epochs=500, lr=0.001, n_per_vowel=300, margin=3.0,
          desat_weight=0.1, l2_weight=1e-4, seed=42):
    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')

    # Start from v4 checkpoint
    ckpt_path = Path('data/m9_substrate_spectral_v4.ckpt.json')
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        W_h = np.array(ckpt['W_h'], dtype=np.float32) * 0.5
        W_x = np.array(ckpt['W_x'], dtype=np.float32) * 0.5
        b = np.array(ckpt['b'], dtype=np.float32)
        print("Initialized from v4 checkpoint (weights scaled to 0.5)")
    else:
        rng = np.random.RandomState(seed)
        scale = 0.05
        W_h = (rng.randn(HIDDEN_DIM, HIDDEN_DIM) * scale).astype(np.float32)
        W_x = (rng.randn(HIDDEN_DIM, 26) * scale).astype(np.float32)
        b = np.zeros(HIDDEN_DIM, dtype=np.float32)
        print("Initialized from random")

    print(f"Generating {n_per_vowel} samples per vowel ({n_per_vowel * len(VOWEL_NAMES)} total)...")
    print(f"Using multiple speaker-mic profiles (clean, 4kHz LP, 3kHz LP, 2kHz LP)...")
    dataset = generate_dataset(n_per_vowel=n_per_vowel, seed=seed)
    print(f"Total: {len(dataset)} samples")
    print(f"Epochs: {n_epochs}, lr: {lr}, margin: {margin}")
    print()

    m_Wh, v_Wh = np.zeros_like(W_h), np.zeros_like(W_h)
    m_Wx, v_Wx = np.zeros_like(W_x), np.zeros_like(W_x)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    best_loss = float('inf')
    best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

    for epoch in range(n_epochs):
        embeddings, labels, caches = [], [], []
        for vowel, audio in dataset:
            h, cache = forward_substrate(W_h, W_x, b, audio, hp_sos)
            embeddings.append(h)
            labels.append(vowel)
            caches.append(cache)
        embeddings = np.array(embeddings, dtype=np.float32)

        tri_loss, d_tri, centroids = compute_triplet_loss_and_grad(embeddings, labels, margin=margin)
        desat_loss, d_desat = compute_desat_loss_and_grad(embeddings, weight=desat_weight)
        l2_loss = l2_weight * (np.sum(W_h ** 2) + np.sum(W_x ** 2) + np.sum(b ** 2))
        total_loss = tri_loss + desat_loss + l2_loss
        d_embeddings = d_tri + d_desat

        d_W_h, d_W_x, d_b = np.zeros_like(W_h), np.zeros_like(W_x), np.zeros_like(b)
        for i in range(len(dataset)):
            dwh_i, dwx_i, db_i = backward_substrate(d_embeddings[i], caches[i], W_h)
            d_W_h += dwh_i; d_W_x += dwx_i; d_b += db_i

        d_W_h += 2 * l2_weight * W_h
        d_W_x += 2 * l2_weight * W_x
        d_b += 2 * l2_weight * b

        flat_grad = np.concatenate([d_W_h.ravel(), d_W_x.ravel(), d_b.ravel()])
        grad_norm = np.linalg.norm(flat_grad)
        if grad_norm > 1.0:
            clip = 1.0 / grad_norm
            d_W_h *= clip; d_W_x *= clip; d_b *= clip

        t_step = epoch + 1
        for pn, g, m, v in [('W_h', d_W_h, m_Wh, v_Wh), ('W_x', d_W_x, m_Wx, v_Wx), ('b', d_b, m_b, v_b)]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            if pn == 'W_h': W_h -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
            elif pn == 'W_x': W_x -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
            else: b -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        if total_loss < best_loss:
            best_loss = total_loss
            best_W_h, best_W_x, best_b = W_h.copy(), W_x.copy(), b.copy()

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            between = [np.linalg.norm(centroids[VOWEL_NAMES[i]] - centroids[VOWEL_NAMES[j]])
                       for i in range(len(VOWEL_NAMES)) for j in range(i+1, len(VOWEL_NAMES))]
            bm = min(between)
            within_max = max(np.mean([np.linalg.norm(embeddings[k] - centroids[v]) for k in range(len(labels)) if labels[k] == v]) for v in VOWEL_NAMES)
            ratio = bm / within_max if within_max > 0 else float('inf')
            print(f"  E{epoch:4d} loss={total_loss:.4f} (tri={tri_loss:.4f}) between_min={bm:.2f} within_max={within_max:.2f} ratio={ratio:.2f}")

    print(f"\nBest loss: {best_loss:.4f}")
    return best_W_h, best_W_x, best_b, dataset


def build_lookup(W_h, W_x, b, dataset, hp_sos, n_augment=200):
    from scipy.spatial import cKDTree
    embeddings = {v: [] for v in VOWEL_NAMES}
    for vowel, audio in dataset:
        h, _ = forward_substrate(W_h, W_x, b, audio, hp_sos)
        embeddings[vowel].append(h)
    rng = np.random.RandomState(123)
    lp_profiles = [None, LP_SOS_4K, LP_SOS_3K, LP_SOS_2K]
    for vowel in VOWEL_NAMES:
        for _ in range(n_augment):
            lp_sos = lp_profiles[rng.randint(len(lp_profiles))]
            audio = gen_vowel(vowel, duration=0.5, f0=rng.uniform(100, 180),
                              jitter=rng.uniform(0.02, 0.08), noise_db=rng.uniform(-45, -30),
                              lp_sos=lp_sos, rng=rng)
            h, _ = forward_substrate(W_h, W_x, b, audio, hp_sos)
            embeddings[vowel].append(h)
    centroids = {v: np.mean(embeddings[v], axis=0).astype(np.float32) for v in VOWEL_NAMES}
    all_h, all_labels = [], []
    for v in VOWEL_NAMES:
        all_h.extend(embeddings[v])
        all_labels.extend([v] * len(embeddings[v]))
    centroid_arr = np.array([centroids[v] for v in VOWEL_NAMES], dtype=np.float32)
    tree = cKDTree(centroid_arr)
    _, idxs = tree.query(np.array(all_h, dtype=np.float32), k=1)
    correct = sum(1 for i, idx in enumerate(idxs) if VOWEL_NAMES[idx] == all_labels[i])
    accuracy = correct / len(all_labels)
    between = [np.linalg.norm(centroids[VOWEL_NAMES[i]] - centroids[VOWEL_NAMES[j]])
               for i in range(len(VOWEL_NAMES)) for j in range(i+1, len(VOWEL_NAMES))]
    within = {}
    for v in VOWEL_NAMES:
        h_arr = np.array(embeddings[v], dtype=np.float32)
        dists_v = np.linalg.norm(h_arr - centroids[v], axis=1)
        within[v] = {'mean': float(np.mean(dists_v)), 'std': float(np.std(dists_v)), 'max': float(np.max(dists_v))}
    print(f"\n=== Lookup Table ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(all_labels)})")
    print(f"Between: {', '.join(f'{d:.2f}' for d in between)} (min={min(between):.2f})")
    for v in VOWEL_NAMES:
        s = within[v]
        print(f"  Within {v}: mean={s['mean']:.2f} std={s['std']:.2f} max={s['max']:.2f}")
    print(f"Separation ratio: {min(between)/max(s['max'] for s in within.values()):.2f}")
    return centroids, within, accuracy


def save_checkpoint(W_h, W_x, b, path):
    data = {'W_h': W_h.tolist(), 'W_x': W_x.tolist(), 'b': b.tolist(),
            'hidden_dim': HIDDEN_DIM, 'frame_size': FRAME, 'sr': SR}
    Path(path).write_text(json.dumps(data))
    print(f"Saved: {path}")


def save_lookup(centroids, within, accuracy, path):
    lookup = {}
    for v in VOWEL_NAMES:
        lookup[v] = {'centroid': centroids[v].tolist(), 'template': VOWEL_FORMANTS[v]}
    data = {'lookup': lookup, 'vowel_names': VOWEL_NAMES,
            'accuracy': float(accuracy), 'within_stats': within}
    Path(path).write_text(json.dumps(data))
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='M9 multi-profile speaker-mic retrain')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-per-vowel', type=int, default=300)
    parser.add_argument('--margin', type=float, default=3.0)
    args = parser.parse_args()

    hp_sos = butter(4, 100, btype='high', fs=SR, output='sos')
    W_h, W_x, b, dataset = train(n_epochs=args.epochs, lr=args.lr,
                                   n_per_vowel=args.n_per_vowel, margin=args.margin)
    save_checkpoint(W_h, W_x, b, 'data/m9_substrate_spectral_v5.ckpt.json')
    centroids, within, accuracy = build_lookup(W_h, W_x, b, dataset, hp_sos)
    save_lookup(centroids, within, accuracy, 'data/m9_formant_lookup_v9.json')
    print("\nDone.")


if __name__ == '__main__':
    main()
