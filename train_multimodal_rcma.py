#!/usr/bin/env python3
"""Train RCMA (or fallback) on the multimodal manifest.

This script is robust: if RCMA cannot be initialized (missing backbone weights),
it falls back to a small test model so you can run smoke tests on cloud.

Usage examples:
  python train_multimodal_rcma.py --manifest manifests/multimodal_manifest.csv --epochs 1 --batch-size 4 --device cpu
"""
import argparse
import os
import sys
from pathlib import Path
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import MultimodalDataset

try:
    from models.model import RCMA
except Exception:
    RCMA = None


def collate_fn(batch):
    # batch is list of samples (dicts)
    # We'll collate video (pad to max T) and audio features similarly
    out = {}
    labels = []

    # Collect keys
    keys = ['video', 'mfcc', 'logmel', 'prosodic']
    for k in keys:
        items = [s[k] for s in batch]
        if all(x is None for x in items):
            out[k] = None
            continue
        # replace None with zeros of first non-None shape
        ref = next(x for x in items if x is not None)
        shapes = [x.shape if x is not None else (0,) for x in items]
        # For simplicity, convert all to tensors and pad/truncate on first dim
        if k == 'video':
            # shapes: (T,C,H,W)
            maxT = max(s[0] if s is not None else 0 for s in items)
            padded = []
            for x in items:
                if x is None:
                    padded.append(torch.zeros((maxT, 1, 48, 48)))
                else:
                    t = x
                    if t.shape[0] < maxT:
                        pad = torch.zeros((maxT - t.shape[0],) + t.shape[1:])
                        t = torch.cat([t, pad], dim=0)
                    else:
                        t = t[:maxT]
                    padded.append(t)
            # shape batch, T, C, H, W -> transpose to batch, C, T, H, W if needed by model
            out[k] = torch.stack(padded)
        else:
            # audio-like: (T, feat)
            maxT = max(s.shape[0] if s is not None else 0 for s in items)
            padded = []
            for x in items:
                if x is None:
                    padded.append(torch.zeros((maxT, ref.shape[1])))
                else:
                    t = x
                    if t.shape[0] < maxT:
                        pad = torch.zeros((maxT - t.shape[0], t.shape[1]))
                        t = torch.cat([t, pad], dim=0)
                    else:
                        t = t[:maxT]
                    padded.append(t)
            out[k] = torch.stack(padded)

    for s in batch:
        labels.append(int(s.get('label', -1)))

    out['label'] = torch.tensor([l - 1 if l > 0 else 0 for l in labels], dtype=torch.long)
    return out


class FallbackModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # simple audio-only classifier that averages temporal frames
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, batch):
        # prefer logmel if present
        x = batch.get('logmel')
        if x is None:
            x = batch.get('mfcc')
        if x is None:
            # try video mean
            v = batch.get('video')
            if v is None:
                # no data
                B = batch['label'].shape[0]
                return torch.zeros((B, 8), device=batch['label'].device)
            # v: B, T, C, H, W or B, C, T, H, W
            if v.dim() == 5:
                # assume B, T, C, H, W -> take mean over spatial + channel
                x = v.mean(dim=[2,3,4])
            else:
                x = v.mean(dim=[1,2,3])

        # x: B, T, F
        x = x.mean(dim=1)
        return self.fc(x)


def try_create_rcma(device, root_dir, visual_ckpt=None, audio_ckpt=None):
    """Try to create RCMA model. Look for checkpoint files in order:
    1. CLI args `visual_ckpt` / `audio_ckpt` (path or stem)
    2. Environment variables `VISUAL_STATE_DICT` / `AUDIO_STATE_DICT`
    3. Search repo for .pth files and pick likely candidates
    Returns model instance or None on failure.
    """
    if RCMA is None:
        return None

    # Helper to transform input (path or stem) to model expected stem
    def stem_from_path(p):
        if not p:
            return ''
        p = str(p)
        if p.endswith('.pth'):
            return Path(p).stem
        return Path(p).stem

    # Prefer CLI args
    visual = visual_ckpt or os.environ.get('VISUAL_STATE_DICT')
    audio = audio_ckpt or os.environ.get('AUDIO_STATE_DICT')

    # If not provided, search for pth files in repo
    if not visual or not audio:
        pth_files = list(Path(root_dir).rglob('*.pth'))
        # heuristics: look for filenames containing 'visual', 'backbone', 'resnet', 'ir', 'vgg'
        def pick_candidate(patterns):
            for p in pth_files:
                name = p.name.lower()
                for pat in patterns:
                    if pat in name:
                        return p
            return None

        if not visual:
            cand = pick_candidate(['visual', 'backbone', 'resnet', 'ir', 'cnn'])
            if cand:
                visual = cand
        if not audio:
            cand = pick_candidate(['vggish', 'audio', 'vgg', 'mfcc'])
            if cand:
                audio = cand

    visual_stem = stem_from_path(visual)
    audio_stem = stem_from_path(audio)

    backbone_settings = {
        'visual_state_dict': visual_stem,
        'audio_state_dict': audio_stem
    }

    try:
        model = RCMA(backbone_settings=backbone_settings, modality=['logmel', 'video'], root_dir=str(root_dir), device=device)
        try:
            model.init()
        except Exception:
            # init may fail if state dicts missing or mismatch; still return model to allow fallback
            pass
        return model
    except Exception:
        return None


def train(args):
    device = torch.device(args.device)
    ds = MultimodalDataset(manifest_path=args.manifest, split='train')
    val_ds = MultimodalDataset(manifest_path=args.manifest, split='val')

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = try_create_rcma(device, ROOT)
    fallback = False
    if model is None:
        print('⚠️  RCMA model unavailable or failed to init — using fallback model for smoke test')
        model = FallbackModel(num_classes=8)
        fallback = True

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val = math.inf

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(dl):
            # move to device
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            # outputs: if seq outputs, reduce
            if outputs.dim() == 3:
                # take mean over time
                outputs = outputs.mean(dim=1)

            loss = criterion(outputs, batch['label'].to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % args.log_interval == 0:
                print(f"Epoch {epoch+1} [{i+1}/{len(dl)}] loss={running_loss/args.log_interval:.4f}")
                running_loss = 0.0

        t1 = time.time()
        print(f"Epoch {epoch+1} completed in {t1-t0:.1f}s")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in vdl:
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                outputs = model(batch)
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
                loss = criterion(outputs, batch['label'].to(device))
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(vdl))
        print(f"Validation loss: {val_loss:.4f}")

        # save checkpoint
        ckpt = Path(args.checkpoint_dir) / f"checkpoint_epoch{epoch+1}.pth"
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt)
        print(f"Saved checkpoint: {ckpt}")

    print('Training finished')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, default='manifests/multimodal_manifest.csv')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p.add_argument('--log-interval', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
