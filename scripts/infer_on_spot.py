#!/usr/bin/env python3
"""Inference helper for Spot: extract audio/frames, load TorchScript, run inference, save CSV."""
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    import torchaudio
except Exception:
    torchaudio = None

# Human-readable label names matching training scripts
LABEL_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful']
# Normalization used during training
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def run_cmd(cmd):
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd)


def extract_audio(video_path: str, out_wav: str):
    run_cmd(['ffmpeg', '-y', '-i', str(video_path), '-vn', '-ac', '1', '-ar', '16000', str(out_wav)])


def extract_frames(video_path: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_cmd(['ffmpeg', '-y', '-i', str(video_path), os.path.join(out_dir, 'frame_%06d.jpg')])


def sample_frame_paths(frames_dir: str, n: int):
    imgs = sorted(Path(frames_dir).glob('*.jpg'))
    if len(imgs) == 0:
        return []
    if len(imgs) <= n:
        return [str(p) for p in imgs]
    idxs = np.linspace(0, len(imgs) - 1, n).astype(int)
    return [str(imgs[i]) for i in idxs]


def load_and_preprocess_frames(frame_paths, img_size=48, device='cpu', normalize=True):
    # Build transform with optional normalization
    transforms_list = [T.Resize((img_size, img_size)), T.ToTensor()]
    if normalize:
        transforms_list.append(T.Normalize(mean=NORM_MEAN, std=NORM_STD))
    tf = T.Compose(transforms_list)
    tensors = []
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        tensors.append(tf(img))
    if len(tensors) == 0:
        return torch.empty(0)
    # Stack to (T, C, H, W) then convert to (1, T, C, H, W)
    arr = torch.stack(tensors, dim=0)  # (T, C, H, W)
    arr = arr.unsqueeze(0).to(device)  # (1, T, C, H, W)

    # Training used exactly 10 frames. Normalize temporal length to 10 by
    # trimming (evenly sampling) or repeating frames when necessary so
    # inference matches training-time shape.
    target_frames = 10
    b, t, c, h, w = arr.shape
    if t == target_frames:
        return arr
    if t > target_frames:
        idxs = np.linspace(0, t - 1, target_frames).astype(int)
        arr = arr[:, idxs, :, :, :].contiguous()
        return arr
    # t < target_frames: repeat frames to reach target length
    repeat_factor = target_frames // t
    remainder = target_frames % t
    parts = [arr.repeat(1, repeat_factor, 1, 1, 1)]
    if remainder > 0:
        parts.append(arr[:, :remainder, :, :, :])
    arr = torch.cat(parts, dim=1).contiguous()
    return arr


def compute_simple_audio_embedding(wav_path: str, device='cpu'):
    if torchaudio is None:
        raise RuntimeError('torchaudio required for audio embedding')
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel = torchaudio.transforms.MelSpectrogram(16000, n_mels=48)(waveform)
    mel = torch.log1p(mel)
    emb = mel.mean(dim=-1).squeeze(0)
    return emb.unsqueeze(0).to(device)


def load_model(path: str, device='cpu'):
    # TorchScript load
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        print('Loaded TorchScript model')
        return m
    except Exception:
        # try torch.load full model
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, torch.nn.Module):
            obj.to(device).eval()
            return obj
        raise RuntimeError('Provide TorchScript or full model object for inference')


def infer_video(model, video, audio, out_csv, device='cpu', frames=30, img_size=48, debug=False, normalize=True):
    tmp = tempfile.mkdtemp(prefix='infer_spot_')
    try:
        if audio is None:
            wav = os.path.join(tmp, 'audio.wav')
            extract_audio(video, wav)
            audio = wav
        frames_dir = os.path.join(tmp, 'frames')
        extract_frames(video, frames_dir)
        frame_paths = sample_frame_paths(frames_dir, frames)
        if not frame_paths:
            print('No frames found')
            return
        video_tensor = load_and_preprocess_frames(frame_paths, img_size=img_size, device=device, normalize=normalize)
        # Validate shape: expect (1, T, C, H, W)
        if video_tensor.dim() != 5:
            raise RuntimeError(f'Unexpected video tensor shape {tuple(video_tensor.shape)}; expected (1, T, C, H, W)')
        # If channels are in the last position (H,W,C) accidentally, try to permute
        b, t, c, h, w = video_tensor.shape
        if c != 3 and w == 3:
            # likely (1, T, H, W, C) -> permute to (1, T, C, H, W)
            try:
                video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
                b, t, c, h, w = video_tensor.shape
                print('Auto-fixed video tensor to channels-first format')
            except Exception:
                raise RuntimeError(f'Cannot fix video tensor shape {tuple(video_tensor.shape)}')
        if c != 3:
            raise RuntimeError(f'Unexpected number of channels: {c}; expected 3')
        # compute audio embedding but many models don't expect it; keep as optional
        try:
            audio_emb = compute_simple_audio_embedding(audio, device=device)
        except Exception:
            audio_emb = None

        model.to(device).eval()
        # Debug info: model summary and parameter stats
        if debug:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                print(f'DEBUG: model total params = {total_params:,}')
                # print first param tensor stats
                for name, p in model.named_parameters():
                    print(f'DEBUG param: {name} shape={tuple(p.shape)} mean={float(p.data.mean()):.6f} std={float(p.data.std()):.6f}')
                    break
            except Exception:
                pass
        with torch.no_grad():
            out = None
            last_err = None
            # Try calling the model with channels-first input (default)
            try:
                out = model(video_tensor)
            except Exception as e1:
                last_err = e1
                # Try channels-last permutation: (B, T, H, W, C)
                try:
                    alt = video_tensor.permute(0, 1, 3, 4, 2).contiguous()
                    print('Retrying model call with channels-last format')
                    out = model(alt)
                    # if successful, record that we permuted
                    video_tensor = alt
                except Exception as e2:
                    last_err = e2
                    # Try passing dict with video/audio (some models expect dict)
                    if audio_emb is not None:
                        try:
                            out = model({'video': video_tensor, 'audio': audio_emb})
                        except Exception as e3:
                            last_err = e3
                    # If still failing, raise the last error
            if out is None and last_err is not None:
                raise last_err

        logits = out
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = logits.cpu().numpy()

        # If debug, save sampled frames and raw logits/probs for inspection
        if debug:
            debug_dir = os.path.join(os.path.dirname(out_csv), Path(out_csv).stem + '_debug')
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            # copy sampled frames
            for i, p in enumerate(frame_paths):
                try:
                    dst = os.path.join(debug_dir, os.path.basename(p))
                    shutil.copy(p, dst)
                except Exception:
                    pass
            # save logits and probs
            try:
                np.save(os.path.join(debug_dir, 'logits.npy'), logits)
                e_d = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs_d = e_d / e_d.sum(axis=1, keepdims=True)
                np.save(os.path.join(debug_dir, 'probs.npy'), probs_d)
                with open(os.path.join(debug_dir, 'summary.txt'), 'w') as fh:
                    fh.write(f'frame_paths={len(frame_paths)}\n')
                    fh.write(f'logits_shape={logits.shape}\n')
                    fh.write(f'probs_min={float(probs_d.min()):.6f} max={float(probs_d.max()):.6f}\n')
            except Exception as e:
                print('DEBUG save error', e)

        # Compute softmax probabilities (stable)
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)

        rows = []
        if logits.shape[0] == 1:
            # Single prediction for the whole clip -> repeat for all frames
            pred = int(np.argmax(probs[0]))
            conf = float(probs[0][pred])
            for p in frame_paths:
                rows.append((os.path.basename(p), pred, conf))
        elif logits.shape[0] == len(frame_paths):
            preds = np.argmax(probs, axis=1)
            for i, p in enumerate(frame_paths):
                pred = int(preds[i])
                conf = float(probs[i][pred])
                rows.append((os.path.basename(p), pred, conf))
        else:
            # Unexpected shape (e.g., per-layer outputs). Average across first axis
            avg_logits = logits.mean(axis=0, keepdims=True)
            e2 = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
            avg_probs = e2 / e2.sum(axis=1, keepdims=True)
            pred = int(np.argmax(avg_probs[0]))
            conf = float(avg_probs[0][pred])
            for p in frame_paths:
                rows.append((os.path.basename(p), pred, conf))

        # Ensure we don't overwrite existing file: if exists, add timestamp
        def make_unique(path):
            p = Path(path)
            if not p.exists():
                return str(p)
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_name = p.with_name(p.stem + '_' + ts + p.suffix)
            return str(new_name)

        out_csv = make_unique(out_csv)

        # Prepare CSV header with per-class probability columns
        prob_cols = [f'p_{n}' for n in LABEL_NAMES]
        header = ['frame', 'pred', 'label', 'conf'] + prob_cols

        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i, (frame_name, pred, conf) in enumerate(rows):
                try:
                    label_name = LABEL_NAMES[int(pred)]
                except Exception:
                    label_name = ''
                # determine per-class probs for this row
                if logits.shape[0] == 1:
                    probs_row = probs[0]
                elif logits.shape[0] == len(frame_paths):
                    probs_row = probs[i]
                else:
                    # averaged case
                    probs_row = avg_probs[0]

                probs_str = [f"{float(x):.4f}" for x in probs_row]
                w.writerow([frame_name, int(pred), label_name, f"{conf:.4f}"] + probs_str)

        print('Wrote', out_csv)

    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--audio', default=None)
    p.add_argument('--out', default='preds.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--frames', type=int, default=10, help='Number of frames to sample (training used 10)')
    p.add_argument('--img-size', type=int, default=48)
    p.add_argument('--debug', action='store_true', help='Save sampled frames and raw logits/probs for debugging')
    p.add_argument('--no-normalize', action='store_true', help='Do not apply ImageNet normalization (use raw [0,1] inputs)')
    args = p.parse_args()

    model = load_model(args.model, device=args.device)
    infer_video(model, args.video, args.audio, args.out, device=args.device, frames=args.frames, img_size=args.img_size, debug=args.debug, normalize=not args.no_normalize)


if __name__ == '__main__':
    main()
