#!/usr/bin/env python3
"""Extract VGGish embeddings for each audio file using the repo's vggish code.

Saves per-file numpy artifacts under `features/audio/{set}/{actor}/{stem}_vggish.npy`.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Workaround for multiple OpenMP runtimes on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Ensure repo root and abaw6_preprocessing are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'abaw6_preprocessing'))

from base.vggish import vggish_params
from base.vggish import vggish_input
from base.vggish.hubconf import vggish as load_vggish

try:
    import soundfile as sf
except Exception as e:
    raise RuntimeError('Please install soundfile (pip install soundfile)')


AUDIO_DIRS = [
    (ROOT / 'Audio_Song_Actors_01-10', 'song'),
    (ROOT / 'Audio_Speech_Actors_11-22', 'speech')
]

OUT_ROOT = ROOT / 'features' / 'audio'
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def save_vggish_for_file(audio_path: Path, model):
    stem = audio_path.stem
    tag = 'song' if 'Audio_Song_Actors_01-10' in str(audio_path) else 'speech'
    actor = audio_path.parent.name
    out_dir = OUT_ROOT / tag / actor
    out_dir.mkdir(parents=True, exist_ok=True)

    # read waveform
    wav_data, sr = sf.read(str(audio_path), dtype='int16')
    # convert to float in [-1,1]
    samples = wav_data.astype(np.float32) / 32768.0

    # convert waveform to log-mel examples expected by vggish
    examples = vggish_input.waveform_to_examples(samples, sr,
                                                vggish_params.EXAMPLE_WINDOW_SECONDS,
                                                vggish_params.EXAMPLE_HOP_SECONDS)

    if examples is None or len(examples) == 0:
        print(f"No vggish examples for {audio_path}")
        return

    # run model (model expects precomputed log-mel examples when preprocess=False)
    import torch
    model.eval()
    with torch.no_grad():
        emb = model(examples, fs=sr)

    emb_np = emb.cpu().numpy()
    np.save(out_dir / f"{stem}_vggish.npy", emb_np)
    print(f"Saved VGGish: {out_dir / (stem + '_vggish.npy')} (shape {emb_np.shape})")


def main():
    # load model without internal preprocessing (we provide precomputed examples) and without postprocess
    model = load_vggish(pretrained=True, preprocess=False, postprocess=False)

    for src_dir, tag in AUDIO_DIRS:
        if not src_dir.exists():
            print(f"Warning: audio dir not found: {src_dir}")
            continue

        for actor_dir in sorted(src_dir.iterdir()):
            if not actor_dir.is_dir():
                continue
            for wav in sorted(actor_dir.glob('*.wav')):
                try:
                    save_vggish_for_file(wav, model)
                except Exception as e:
                    print(f"Error processing {wav}: {e}")

    print('VGGish extraction complete.')


if __name__ == '__main__':
    main()
