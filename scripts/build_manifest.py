#!/usr/bin/env python3
"""Build a multimodal manifest CSV using the repo's RAVDESSDataset handler.

Produces `manifests/multimodal_manifest.csv` with paired entries and feature paths.
"""
import os
import sys
import csv
from pathlib import Path

# Workaround for multiple OpenMP runtimes on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'abaw6_preprocessing'))

from ravdess_handler import RAVDESSDataset

OUT_DIR = ROOT / 'manifests'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'manifests' / 'multimodal_manifest.csv' if False else ROOT / 'manifests' / 'multimodal_manifest.csv'

FEATURE_AUDIO_ROOT = ROOT / 'features' / 'audio'
FEATURE_VIDEO_ROOT = ROOT / 'features' / 'video'

def feature_paths_for_audio(audio_path):
    # derive paths based on naming created by extract_audio_features.py
    fn = Path(audio_path).stem
    tag = 'song' if 'Audio_Song_Actors_01-10' in str(audio_path) else 'speech'
    actor = Path(audio_path).parent.name
    base = FEATURE_AUDIO_ROOT / tag / actor
    return {
        'mfcc': str(base / f"{fn}_mfcc.npy"),
        'logmel': str(base / f"{fn}_logmel.npy"),
        'pros': str(base / f"{fn}_pros.npy")
    }

def feature_path_for_video(video_path):
    fn = Path(video_path).stem
    tag = 'song' if 'Video_Song_Actors_01-10' in str(video_path) else 'speech'
    actor = Path(video_path).parent.name
    base = FEATURE_VIDEO_ROOT / tag / actor
    return str(base / f"{fn}_video.npy")

def build():
    ds = RAVDESSDataset()
    info = ds.create_dataset_info()

    paired = info.get('paired_files', [])

    # create simple actor-based split (train/val/test by actor index)
    actors = sorted(list(info.get('actors', [])))
    actor_to_split = {}
    for i, a in enumerate(actors):
        r = i % 10
        if r < 7:
            actor_to_split[a] = 'train'
        elif r < 9:
            actor_to_split[a] = 'val'
        else:
            actor_to_split[a] = 'test'

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'set', 'actor', 'emotion', 'emotion_label', 'video_path', 'audio_path',
            'video_feat', 'mfcc', 'logmel', 'prosodic', 'split'
        ])
        writer.writeheader()

        for p in paired:
            video_p = p['video']['path']
            audio_p = p['audio']['path']
            actor = p.get('actor', '')
            emo = p.get('emotion', '')
            emo_label = p.get('emotion_label', '')

            vfeat = feature_path_for_video(video_p)
            afeat = feature_paths_for_audio(audio_p)

            split = actor_to_split.get(actor, 'train')
            set_tag = 'song' if 'Song' in video_p else 'speech'

            writer.writerow({
                'set': set_tag,
                'actor': actor,
                'emotion': emo,
                'emotion_label': emo_label,
                'video_path': video_p,
                'audio_path': audio_p,
                'video_feat': vfeat,
                'mfcc': afeat['mfcc'],
                'logmel': afeat['logmel'],
                'prosodic': afeat['pros'],
                'split': split
            })

    print(f"Manifest written: {OUT_CSV} (pairs: {len(paired)})")


if __name__ == '__main__':
    build()
