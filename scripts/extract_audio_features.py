#!/usr/bin/env python3
"""Extract audio features (mfcc, logmel, prosodic) using repo's AudioProcessor

Saves per-file numpy artifacts under `features/audio/{set}/{actor}/{stem}_{feat}.npy`
"""
import sys
import os
from pathlib import Path
import numpy as np

# Workaround for multiple OpenMP runtimes on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Ensure repo root is on sys.path so imports work when run from different shells
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'abaw6_preprocessing'))

from multimodal_processor import AudioProcessor


ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIRS = [
    (ROOT / 'Audio_Song_Actors_01-10', 'song'),
    (ROOT / 'Audio_Speech_Actors_11-22', 'speech')
]

OUT_ROOT = ROOT / 'features' / 'audio'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def save_features(audio_path: Path, proc: AudioProcessor, target_length=100):
    rel_parts = audio_path.parts[audio_path.parts.index('Audio_Song_Actors_01-10')+1:] if 'Audio_Song_Actors_01-10' in audio_path.parts else audio_path.parts[audio_path.parts.index('Audio_Speech_Actors_11-22')+1:]
    actor = rel_parts[0] if rel_parts else 'unknown'
    stem = audio_path.stem

    out_dir = OUT_ROOT / ('song' if 'Audio_Song_Actors_01-10' in audio_path.parts else 'speech') / actor
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = proc.process_audio(audio_path, target_length=target_length)

    mfcc = feats['mfcc'].numpy()
    logmel = feats['logmel'].numpy()
    pros = feats['prosodic'].numpy()

    np.save(out_dir / f"{stem}_mfcc.npy", mfcc)
    np.save(out_dir / f"{stem}_logmel.npy", logmel)
    np.save(out_dir / f"{stem}_pros.npy", pros)

    print(f"Saved: {out_dir / (stem + '_mfcc.npy')} (mfcc {mfcc.shape})")


def main():
    # Use 16kHz for compatibility with VGGish and Spot deployment
    proc = AudioProcessor(sample_rate=16000)

    for src_dir, tag in AUDIO_DIRS:
        if not src_dir.exists():
            print(f"Warning: audio dir not found: {src_dir}")
            continue

        for actor_dir in sorted(src_dir.iterdir()):
            if not actor_dir.is_dir():
                continue
            for wav in sorted(actor_dir.glob('*.wav')):
                try:
                    save_features(wav, proc, target_length=100)
                except Exception as e:
                    print(f"Error processing {wav}: {e}")

    print("Audio extraction complete.")


if __name__ == '__main__':
    main()
