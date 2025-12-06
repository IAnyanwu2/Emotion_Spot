#!/usr/bin/env python3
"""Generate missing `vggish.npy` embeddings for trials under a processed_trials tree.

Behavior:
- Walk `--dataset-root` (default: `/opt/dataset/processed_trials`) and for each trial folder
  that lacks `vggish.npy` try, in order:
  1. Find a raw WAV whose stem contains the trial id under common audio folders and compute embeddings.
  2. If `logmel.npy` exists in the trial folder and looks like VGGish examples, run the model on it.
  3. If a precomputed `*_vggish.npy` exists under `features/audio/...`, copy it into the trial folder.

This is a best-effort helper; adjust paths/roots for your environment before running.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import shutil
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='/opt/dataset/processed_trials', help='Root containing processed trial folders (default /opt/dataset/processed_trials)')
    parser.add_argument('--repo-root', default='/opt/repo', help='Path to repo root where vggish code may live')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    DATASET_ROOT = Path(args.dataset_root)
    REPO_ROOT = Path(args.repo_root)

    # Make sure repo and abaw6_preprocessing are on sys.path for local vggish
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / 'abaw6_preprocessing'))

    try:
        from base.vggish import vggish_input, vggish_params
        from base.vggish.hubconf import vggish as load_vggish
    except Exception as e:
        print('Could not import local vggish module from repo. Ensure `base/vggish` or `abaw6_preprocessing/base/vggish` exists.')
        raise

    # If a local pretrained file exists at /opt/pretrained/vggish.pth, copy it
    # into torch's hub checkpoints path expected by the vggish loader. This
    # avoids network downloads and ensures the local weights are used.
    local_weights = Path('/opt/pretrained/vggish.pth')
    if local_weights.exists():
        hub_dir = Path(os.path.expanduser('~')) / '.cache' / 'torch' / 'hub' / 'checkpoints'
        hub_dir.mkdir(parents=True, exist_ok=True)
        target = hub_dir / 'vggish-10086976.pth'
        try:
            shutil.copy2(str(local_weights), str(target))
            print(f'Copied local vggish weights to {target}')
        except Exception as e:
            print(f'Warning: could not copy local vggish weights: {e}')

    try:
        import soundfile as sf
    except Exception:
        print('Please install soundfile in the venv: pip install soundfile')
        raise

    model = load_vggish(pretrained=True, preprocess=False, postprocess=False)
    model.eval()

    # Walk dataset tree: expecting structure like /.../processed_trials/<modality>/<tag>/<actor>/<trial>
    trial_dirs = [p for p in DATASET_ROOT.rglob('*') if p.is_dir() and (p / 'logmel.npy').exists() or (p / 'mfcc.npy').exists() or (p / 'prosodic.npy').exists()]

    # Fallback: collect directories that are direct children at depth 4 (modality/tag/actor/trial)
    if not trial_dirs:
        trial_dirs = [p for p in DATASET_ROOT.glob('**/*') if p.is_dir() and len(list(p.iterdir())) > 0]

    print(f'Found {len(trial_dirs)} candidate trial dirs (may include non-trial folders)')

    for trial in sorted(trial_dirs):
        vgg_path = trial / 'vggish.npy'
        if vgg_path.exists():
            continue

        trial_name = trial.name
        actor = trial.parent.name if trial.parent else ''

        # 1) Try to find precomputed *_vggish.npy in features area
        candidates = list((REPO_ROOT / 'features').rglob(f'*{trial_name}*vggish.npy'))
        candidates += list((REPO_ROOT / 'features').rglob(f'*{trial_name}*vggish.npy'))
        # also check dataset features
        candidates += list((DATASET_ROOT.parent / 'features').rglob(f'*{trial_name}*vggish.npy'))

        if candidates:
            src = candidates[0]
            if args.dry_run:
                print(f'[DRY] Would copy precomputed {src} -> {vgg_path}')
            else:
                vgg_path.write_bytes(src.read_bytes())
                print(f'Copied precomputed vggish: {src} -> {vgg_path}')
            continue

        # 2) Try to find a raw WAV matching the trial stem under common audio folders
        wav_candidates = []
        common_audio_roots = [REPO_ROOT / 'Audio_Song_Actors_01-10', REPO_ROOT / 'Audio_Speech_Actors_11-22', DATASET_ROOT.parent / 'features' / 'audio']
        for root in common_audio_roots:
            if not root.exists():
                continue
            for p in root.rglob('*.wav'):
                if trial_name in p.stem or actor in str(p.parent):
                    wav_candidates.append(p)

        if wav_candidates:
            wav = wav_candidates[0]
            if args.dry_run:
                print(f'[DRY] Would run vggish on {wav} -> {vgg_path}')
                continue
            print(f'Processing WAV {wav} for trial {trial}')
            try:
                wav_data, sr = sf.read(str(wav), dtype='int16')
            except Exception as e:
                print(f'Failed reading {wav}: {e}')
                continue
            samples = wav_data.astype(np.float32) / 32768.0
            examples = vggish_input.waveform_to_examples(samples, sr,
                                                        vggish_params.EXAMPLE_WINDOW_SECONDS,
                                                        vggish_params.EXAMPLE_HOP_SECONDS)
            if examples is None or len(examples) == 0:
                print(f'No vggish examples for {wav}')
                continue
            import torch
            with torch.no_grad():
                emb = model(examples, fs=sr)
            emb_np = emb.cpu().numpy()
            np.save(vgg_path, emb_np)
            print(f'Saved vggish embeddings to {vgg_path} (shape {emb_np.shape})')
            continue

        # 3) Try to use logmel.npy inside the trial directory
        logmel = trial / 'logmel.npy'
        if logmel.exists():
            if args.dry_run:
                print(f'[DRY] Would run vggish on logmel {logmel} -> {vgg_path}')
                continue
            try:
                examples = np.load(logmel, mmap_mode=None)
            except Exception as e:
                print(f'Failed loading {logmel}: {e}')
                continue
            # Expect examples to be shaped like (N, 96, 64) or similar
                try:
                    # Ensure examples is 3-D: (num_examples, num_frames, num_bands)
                    if isinstance(examples, np.ndarray) and examples.ndim == 2:
                        examples = examples[np.newaxis, ...]
                    if not (isinstance(examples, np.ndarray) and examples.ndim == 3):
                        print(f'logmel exists but has unexpected shape {getattr(examples, "shape", None)}; skipping')
                        continue

                    import torch
                    with torch.no_grad():
                        emb = model(examples, fs=None)
                    emb_np = emb.cpu().numpy()
                    np.save(vgg_path, emb_np)
                    print(f'Computed vggish from logmel: {vgg_path} (shape {emb_np.shape})')
                    continue
                except Exception as e:
                    print(f'logmel exists but could not be used as vggish input: {e}')

        print(f'Could not find audio/logmel/vggish for trial {trial}; skipping')


if __name__ == '__main__':
    main()
