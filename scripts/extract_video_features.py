
#!/usr/bin/env python3
"""Extract video frame features using repo's VideoProcessor

Saves per-file numpy artifacts under `features/video/{set}/{actor}/{stem}_video.npy` as array (T, C, H, W)
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

from multimodal_processor import VideoProcessor

VIDEO_DIRS = [
    (ROOT / 'Video_Song_Actors_01-10', 'song'),
    (ROOT / 'Video_Speech_Actors_11-22', 'speech')
]

OUT_ROOT = ROOT / 'features' / 'video'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def save_video(video_path: Path, proc: VideoProcessor, max_frames=100):
    # determine actor and tag
    if 'Video_Song_Actors_01-10' in video_path.parts:
        tag = 'song'
        idx = video_path.parts.index('Video_Song_Actors_01-10')
    else:
        tag = 'speech'
        idx = video_path.parts.index('Video_Speech_Actors_11-22')

    rel_parts = video_path.parts[idx+1:]
    actor = rel_parts[0] if rel_parts else 'unknown'
    stem = video_path.stem

    out_dir = OUT_ROOT / tag / actor
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = proc.extract_frames(video_path, max_frames=max_frames)

    # ensure numpy array
    arr = frames.numpy()
    np.save(out_dir / f"{stem}_video.npy", arr)

    print(f"Saved: {out_dir / (stem + '_video.npy')} (frames {arr.shape})")


def main():
    proc = VideoProcessor(target_size=(48,48), fps_target=10)

    for src_dir, tag in VIDEO_DIRS:
        if not src_dir.exists():
            print(f"Warning: video dir not found: {src_dir}")
            continue

        for actor_dir in sorted(src_dir.rglob('*')):
            # allow either Actor_XX folders or nested structure
            if actor_dir.is_dir():
                for video in sorted(actor_dir.glob('*.mp4')):
                    try:
                        save_video(video, proc, max_frames=100)
                    except Exception as e:
                        print(f"Error processing {video}: {e}")

    print("Video extraction complete.")


if __name__ == '__main__':
    main()
