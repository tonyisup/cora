#!/usr/bin/env python3
"""M9 Live Vowel Imitation — spectral substrate with feedback prevention.

Uses Mel-filterbank spectral features as input to the substrate RNN,
giving direct access to F1/F2/F3 energy ratios that cleanly separate
vowels even after normalization.

Run:  python scripts/m9_live_vowel.py
"""

import sys, json, time
from collections import Counter
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.spatial import cKDTree
sys.path.insert(0, 'packages')
import sounddevice as sd

SR = 16000
FRAME = 480
N_MEL = 26

# ── Mel filterbank (same as m9_train_spectral.py) ─────────────────────────

def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def make_mel_filterbank(n_mels=N_MEL, n_fft=512, sr=SR, fmin=80, fmax=7600):
    n_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bin_points = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m+1], bin_points[m+2]
        for k in range(left, center):
            if k < n_bins and center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center:
                fb[m, k] = (right - k) / (right - center)
    return fb

MEL_FB = make_mel_filterbank()

def compute_mel(frame, mel_fb=MEL_FB, n_fft=512):
    x = np.zeros(n_fft, dtype=np.float32)
    n = min(len(frame), n_fft)
    x[:n] = frame[:n]
    X = np.abs(np.fft.rfft(x))
    mel_spec = mel_fb @ X
    mel_spec = np.log(mel_spec + 1e-6).astype(np.float32)
    rms = np.sqrt(np.mean(mel_spec ** 2))
    if rms > 1e-6:
        mel_spec /= rms
    return mel_spec

# ── RNN forward (inline, matches training) ────────────────────────────────

HIDDEN_DIM = 32

# Load spectral substrate
sub_ckt = json.load(open('data/m9_substrate_spectral.ckpt.json'))
W_h = np.array(sub_ckt['W_h'], dtype=np.float32)
W_x = np.array(sub_ckt['W_x'], dtype=np.float32)
b = np.array(sub_ckt['b'], dtype=np.float32)

# Load lookup (v5 — spectral)
lookup_data = json.load(open('data/m9_formant_lookup_v5.json'))
vowel_names = lookup_data['vowel_names']
lookup = lookup_data['lookup']
centroids = np.array([lookup[v]['centroid'] for v in vowel_names], dtype=np.float32)
tree = cKDTree(centroids)

hp = butter(4, 100, btype='high', fs=SR, output='sos')

def synth(vowel, n_frames, phase=0.0):
    t = lookup[vowel]['template']
    frames, cp = [], phase
    for _ in range(n_frames):
        tf = np.arange(480, dtype=np.float32) / SR
        fr = np.zeros(480, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            fr += t[ak] * np.sin(2*np.pi*t[fk]*tf + cp)
            cp += 2*np.pi*t[fk]*(480/SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0: fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2*np.pi
    return np.concatenate(frames), cp

def synth_interp(template_from, template_to, alpha, n_frames, phase=0.0):
    """Synthesize frames with interpolated formant parameters.

    alpha=0.0 → template_from, alpha=1.0 → template_to.
    Good for smooth vowel transitions (~60-90ms).
    """
    frames, cp = [], phase
    for _ in range(n_frames):
        tf = np.arange(480, dtype=np.float32) / SR
        fr = np.zeros(480, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            f_interp = template_from[fk] * (1 - alpha) + template_to[fk] * alpha
            a_interp = template_from[ak] * (1 - alpha) + template_to[ak] * alpha
            fr += a_interp * np.sin(2*np.pi*f_interp*tf + cp)
            cp += 2*np.pi*f_interp*(480/SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0: fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2*np.pi
    return np.concatenate(frames), cp

def main():
    import argparse
    parser = argparse.ArgumentParser(description='M9 Live Vowel Imitation (Spectral)')
    parser.add_argument('--input-device', type=int, default=1, help='Mic input device')
    parser.add_argument('--output-device', type=int, default=2, help='Headphone output device')
    parser.add_argument('--threshold', type=float, default=0.003, help='RMS threshold for speech')
    parser.add_argument('--feedback-frames', type=int, default=5,
                        help='Frames to ignore input after producing output')
    parser.add_argument('--hold-frames', type=int, default=3, help='Frames to hold vowel')
    parser.add_argument('--smooth-frames', type=int, default=3,
                        help='Majority-vote window for temporal smoothing')
    parser.add_argument('--transition-frames', type=int, default=3,
                        help='Frames to interpolate formants during vowel change (~90ms)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Min centroid distance ratio for acceptance (0=accept all)')
    args = parser.parse_args()

    print(f"Cora Vowel Imitation (Spectral)")
    print(f"  Input: device {args.input_device} (mic)")
    print(f"  Output: device {args.output_device} (headphones)")
    print(f"  Vowels: {vowel_names}")
    print(f"  Mel bins: {N_MEL}")
    print(f"  Feedback gate: {args.feedback_frames} frames ({args.feedback_frames * 30}ms)")
    print(f"  Temporal smoothing: {args.smooth_frames}-frame majority vote")
    print(f"  Vowel transitions: {args.transition_frames}-frame formant interpolation ({args.transition_frames * 30}ms)")
    print(f"\nSpeak clearly. Hold each vowel 0.5s+. Ctrl+C to stop.\n")

    # Pre-compute distances between centroids for confidence scoring
    centroid_dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)

    state = {
        'prev': None,
        'phase': 0.0,
        'hold': 0,
        'silence_count': args.feedback_frames,
        'history': [],
        'h': np.zeros(HIDDEN_DIM, dtype=np.float32),  # RNN hidden state
        'transition': 0,      # Frames remaining in vowel transition
        'transition_from': None,  # Template we're transitioning FROM
        'transition_to': None,    # Template we're transitioning TO
    }

    def rnn_step(mel_features):
        """One RNN step: h = tanh(W_h @ h + W_x @ mel + b)."""
        state['h'] = np.tanh(W_h @ state['h'] + W_x @ mel_features + b)

    def callback(indata, outdata, frames, time_info, status):
        mic = sosfilt(hp, indata[:, 0].astype(np.float32))
        rms = np.sqrt(np.mean(mic ** 2))

        out_v = None

        if rms > args.threshold and state['silence_count'] >= args.feedback_frames:
            # Normalize block RMS to match training audio (~0.3 RMS after HP filter)
            # This ensures Mel features have the same scale as training
            block_rms = np.sqrt(np.mean(mic ** 2))
            mic_scaled = mic / max(block_rms, 1e-6) * 0.3

            # Compute Mel features for each frame — only feed speech frames
            # (skip silence frames that would corrupt the RNN state)
            # Reset RNN for each new utterance
            state['h'] = np.zeros(HIDDEN_DIM, dtype=np.float32)
            n_speech_frames = 0

            for i in range(len(mic_scaled) // FRAME):
                raw_frame = mic_scaled[i*FRAME:(i+1)*FRAME].flatten()
                frame_rms = np.sqrt(np.mean(raw_frame ** 2))
                if frame_rms > args.threshold:
                    mel_features = compute_mel(raw_frame)
                    rnn_step(mel_features)
                    n_speech_frames += 1

            # Skip classification if not enough speech frames in this block
            if n_speech_frames < 2:
                state['silence_count'] = min(state['silence_count'] + 1,
                                             args.feedback_frames * 2)
                out_v = None
            else:
                h = state['h']

                # Classify via nearest centroid
                dists, idx = tree.query(h, k=2)  # Get 2 nearest for confidence
                v_raw = vowel_names[idx[0]]

                # Confidence: ratio of dist to 2nd-nearest vs 1st-nearest
                # Higher = more confident (1st choice is much closer than 2nd)
                if len(idx) > 1 and dists[0] > 1e-6:
                    confidence = dists[1] / dists[0]  # >1 means confident
                else:
                    confidence = float('inf')

                # Temporal smoothing: majority vote
                state['history'].append(v_raw)
                if len(state['history']) > args.smooth_frames:
                    state['history'] = state['history'][-args.smooth_frames:]
                counts = Counter(state['history'])
                v = counts.most_common(1)[0][0]

                # Skip if low confidence (optional)
                if args.confidence_threshold > 0 and confidence < args.confidence_threshold:
                    # Low confidence — don't change output
                    if state['prev'] is not None:
                        out_v = state['prev']
                        state['hold'] = args.hold_frames
                else:
                    out_v = v
                    state['hold'] = args.hold_frames

                state['silence_count'] = 0

                if v != state['prev']:
                    conf_str = f" (conf={confidence:.1f})" if confidence < float('inf') else ""
                    print(f"  {v}{conf_str}", end='\r', flush=True)
                state['prev'] = v

        elif state['hold'] > 0:
            state['hold'] -= 1
            out_v = state['prev']
            state['silence_count'] = 0
        else:
            state['silence_count'] = min(state['silence_count'] + 1, args.feedback_frames * 2)
            if state['silence_count'] >= args.feedback_frames:
                state['history'] = []

        # Produce audio output (with formant interpolation during transitions)
        if out_v:
            n_out = max(1, len(mic) // FRAME)
            # Check if we should start a new transition
            if out_v != state['prev'] and state['prev'] is not None and args.transition_frames > 0:
                state['transition'] = args.transition_frames
                state['transition_from'] = lookup[state['prev']]['template']
                state['transition_to'] = lookup[out_v]['template']

            if state['transition'] > 0 and state['transition_from'] is not None:
                # In transition — interpolate formants
                alpha = 1.0 - (state['transition'] / args.transition_frames)
                audio, state['phase'] = synth_interp(
                    state['transition_from'], state['transition_to'],
                    alpha, n_out, state['phase'])
                state['transition'] -= 1
            else:
                # Steady vowel — use normal synth
                audio, state['phase'] = synth(out_v, n_out, state['phase'])

            n = len(outdata)
            if len(audio) >= n:
                outdata[:] = audio[:n].reshape(-1, 1)
            else:
                outdata[:len(audio)] = audio.reshape(-1, 1)
                outdata[len(audio):] = 0
        else:
            outdata[:] = 0

    try:
        sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=SR, blocksize=SR // 2, channels=(1, 1),
            callback=callback, dtype='float32'
        ).__enter__()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == '__main__':
    main()
