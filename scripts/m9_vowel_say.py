#!/usr/bin/env python3
"""M9 Text → Vowel Association.

Type a word and hear its vowel imitated. Uses a pronunciation dictionary
to map words to vowel sequences, then synthesizes through the spectral
substrate pipeline (Mel features → RNN → centroid lookup → formant synth).

Two modes:
  1. Interactive: type words, hear vowels. Ctrl+C to stop.
  2. Single: --word <word> plays the vowel once.

Usage:
    python scripts/m9_vowel_say.py
    python scripts/m9_vowel_say.py --word hello
"""

import sys, json, time
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.spatial import cKDTree
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'packages'))
import sounddevice as sd

SR = 16000
FRAME = 480
N_MEL = 26
HIDDEN_DIM = 32

# ── Pronunciation dictionary: word → vowel sequence ──────────────────────

# Maps common English words to their vowel content using the 5-vowel set:
# /i/ (beat), /ɛ/ (bet), /ɑ/ (hot), /ɔ/ (caught), /u/ (boot)
PRONUNCIATION = {
    # /i/ words
    'beat': 'i', 'see': 'i', 'me': 'i', 'she': 'i', 'be': 'i',
    'feed': 'i', 'leave': 'i', 'deep': 'i', 'keep': 'i', 'need': 'i',
    'feel': 'i', 'three': 'i', 'green': 'i', 'teeth': 'i', 'key': 'i',
    # /ɛ/ words
    'bet': 'E', 'red': 'E', 'get': 'E', 'let': 'E', 'said': 'E',
    'head': 'E', 'well': 'E', 'then': 'E', 'went': 'E', 'tell': 'E',
    'help': 'E', 'best': 'E', 'check': 'E', 'step': 'E', 'yes': 'E',
    # /ɑ/ words
    'hot': 'a', 'not': 'a', 'got': 'a', 'stop': 'a', 'lot': 'a',
    'box': 'a', 'top': 'a', 'drop': 'a', 'dog': 'a', 'clock': 'a',
    'block': 'a', 'shop': 'a', 'lost': 'a', 'rock': 'a', 'sock': 'a',
    # /ɔ/ words
    'caught': 'O', 'bought': 'O', 'thought': 'O', 'walk': 'O', 'talk': 'O',
    'all': 'O', 'call': 'O', 'ball': 'O', 'fall': 'O', 'tall': 'O',
    'small': 'O', 'law': 'O', 'saw': 'O', 'raw': 'O', 'draw': 'O',
    # /u/ words
    'boot': 'u', 'food': 'u', 'too': 'u', 'move': 'u', 'room': 'u',
    'blue': 'u', 'true': 'u', 'school': 'u', 'choose': 'u', 'soon': 'u',
    'moon': 'u', 'loop': 'u', 'tube': 'u', 'rule': 'u', 'shoe': 'u',
    # Multi-vowel words
    'hello': 'E-u', 'goodbye': 'u-i', 'morning': 'O-i',
    'evening': 'i-i', 'welcome': 'E-E', 'python': 'i-O',
    'audio': 'O-i-u', 'video': 'i-E-u', 'music': 'u-i',
    'hello world': 'E-u-O', 'cora': 'O-a',
}

# ── Mel filterbank ────────────────────────────────────────────────────────

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
            if k < n_bins and center > left: fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if k < n_bins and right > center: fb[m, k] = (right - k) / (right - center)
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
    if rms > 1e-6: mel_spec /= rms
    return mel_spec

# ── Load substrate + lookup ───────────────────────────────────────────────

sub_ckt = json.load(open('data/m9_substrate_spectral.ckpt.json'))
W_h = np.array(sub_ckt['W_h'], dtype=np.float32)
W_x = np.array(sub_ckt['W_x'], dtype=np.float32)
b_vec = np.array(sub_ckt['b'], dtype=np.float32)

lookup_data = json.load(open('data/m9_formant_lookup_v5.json'))
vowel_names = lookup_data['vowel_names']
lookup = lookup_data['lookup']
centroids = np.array([lookup[v]['centroid'] for v in vowel_names], dtype=np.float32)

VOWEL_IPA = {'i': '/i/', 'E': '/ɛ/', 'a': '/ɑ/', 'O': '/ɔ/', 'u': '/u/'}

hp = butter(4, 100, btype='high', fs=SR, output='sos')

# ── Synthesis ─────────────────────────────────────────────────────────────

def synth_vowel(vowel, duration=0.4, phase=0.0):
    """Synthesize a vowel using its formant template. Returns audio + final phase."""
    t = lookup[vowel]['template']
    n_frames = int(duration * SR / FRAME)
    frames = []
    cp = phase
    for _ in range(n_frames):
        tf = np.arange(FRAME, dtype=np.float32) / SR
        fr = np.zeros(FRAME, dtype=np.float32)
        for fk, ak in [('f1','a1'),('f2','a2'),('f3','a3')]:
            fr += t[ak] * np.sin(2*np.pi*t[fk]*tf + cp)
            cp += 2*np.pi*t[fk]*(FRAME/SR)
        pk = np.max(np.abs(fr))
        if pk > 1.0: fr /= pk
        frames.append(fr.astype(np.float32))
        cp %= 2*np.pi
    return np.concatenate(frames), cp

def synth_vowel_sequence(vowels, duration_each=0.4):
    """Synthesize a sequence of vowels with smooth phase continuity."""
    parts = []
    phase = 0.0
    for v in vowels:
        if v not in lookup:
            print(f"  Unknown vowel: {v}")
            continue
        audio, phase = synth_vowel(v, duration=duration_each, phase=phase)
        parts.append(audio)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)

# ── Word lookup ───────────────────────────────────────────────────────────

def word_to_vowels(word):
    """Map a word to its vowel sequence. Returns list of vowel codes."""
    w = word.lower().strip()
    if w in PRONUNCIATION:
        return PRONUNCIATION[w].split('-')
    # Try to guess: check each character against vowels
    # (simple fallback for single vowels typed directly)
    if w in vowel_names:
        return [w]
    if len(w) == 1 and w in 'ieaou':
        mapping = {'i': 'i', 'e': 'E', 'a': 'a', 'o': 'O', 'u': 'u'}
        return [mapping.get(w, w)]
    return None

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='M9 Text → Vowel Association')
    parser.add_argument('--word', type=str, help='Single word to say')
    parser.add_argument('--output-device', type=int, default=2, help='Audio output device')
    parser.add_argument('--duration', type=float, default=0.4, help='Duration per vowel (s)')
    parser.add_argument('--list', action='store_true', help='List known words')
    args = parser.parse_args()

    if args.list:
        print("Known words by vowel:")
        for v in vowel_names:
            words = [w for w, vs in PRONUNCIATION.items() if vs.startswith(v) and '-' not in vs]
            ipa = VOWEL_IPA.get(v, v)
            print(f"  {ipa}: {', '.join(sorted(words))}")
        print("\nMulti-vowel words:")
        multi = {w: vs for w, vs in PRONUNCIATION.items() if '-' in vs}
        for w, vs in sorted(multi.items()):
            vs_ipa = ' → '.join(VOWEL_IPA.get(v, v) for v in vs.split('-'))
            print(f"  {w}: {vs_ipa}")
        return

    if args.word:
        vowels = word_to_vowels(args.word)
        if vowels is None:
            print(f"Unknown word: {args.word}")
            print(f"Type a known word, a vowel code (i E a O u), or use --list to see all.")
            return
        vs_ipa = ' → '.join(VOWEL_IPA.get(v, v) for v in vowels)
        print(f"{args.word} → {vs_ipa}")
        audio = synth_vowel_sequence(vowels, duration_each=args.duration)
        if len(audio) > 0:
            sd.play(audio, samplerate=SR, device=args.output_device, blocking=True)
        return

    # Interactive mode
    print("Cora Text → Vowel")
    print(f"  Vowels: {', '.join(VOWEL_IPA[v] for v in vowel_names)}")
    print(f"  Type a word to hear its vowel. Enter to skip. Ctrl+C to stop.")
    print(f"  Use --list to see all known words.")
    print()

    while True:
        try:
            word = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not word:
            continue

        if word == 'list':
            # Inline list
            for v in vowel_names:
                words = [w for w, vs in PRONUNCIATION.items() if vs.startswith(v) and '-' not in vs]
                print(f"    {VOWEL_IPA.get(v, v)}: {', '.join(sorted(words)[:8])}...")
            continue

        vowels = word_to_vowels(word)
        if vowels is None:
            print(f"  Unknown: {word}")
            continue

        vs_ipa = ' → '.join(VOWEL_IPA.get(v, v) for v in vowels)
        audio = synth_vowel_sequence(vowels, duration_each=args.duration)

        if len(audio) > 0:
            print(f"  {word} → {vs_ipa}")
            sd.play(audio, samplerate=SR, device=args.output_device, blocking=True)
        time.sleep(0.1)


if __name__ == '__main__':
    main()
