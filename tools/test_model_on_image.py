#!/usr/bin/env python3
"""Compare TorchScript vs checkpoint outputs on a single sample image.

Usage:
    python tools/test_model_on_image.py --img <path> --model_ts <path> --ckpt <path>
"""
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True)
parser.add_argument('--model_ts', required=True)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--model-class', default='massive', choices=['massive','spot'])
args = parser.parse_args()

img = Image.open(args.img).convert('RGB').resize((48,48))

if args.model_class == 'massive':
    tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    from train_multimodal_massive import MultimodalEmotionCNN
    m = MultimodalEmotionCNN(num_classes=6, input_size=48)
else:
    # Spot model
    tf = T.Compose([T.ToTensor()])
    from train_spot_emotion import SimpleEmotionCNN
    m = SimpleEmotionCNN(num_classes=6 if args.model_class == 'massive' else 8, input_size=48)

x = tf(img).unsqueeze(0)  # (1, C, H, W)
# Video shape expected: (1, T, C, H, W) for massive, or (1, T, H, W, C) for spot
x_video = x.unsqueeze(0).repeat(1, 10, 1, 1, 1)

# TorchScript
ts = torch.jit.load(args.model_ts, map_location='cpu').eval()
with torch.no_grad():
    out_ts = ts(x_video)
    out_ts_np = out_ts.detach().cpu().numpy()

# Checkpoint
sd = torch.load(args.ckpt, map_location='cpu')
try:
    m.load_state_dict(sd)
except Exception:
    # If saved as full model object, try load as is
    if isinstance(sd, torch.nn.Module):
        m = sd
    else:
        # sd might be a state dict; we'll try
        m.load_state_dict(sd)
m.eval()
with torch.no_grad():
    out_ckpt = m(x_video)
    out_ckpt_np = out_ckpt.detach().cpu().numpy()

print('TorchScript logits:', out_ts_np.shape, out_ts_np)
print('Checkpoint logits:', out_ckpt_np.shape, out_ckpt_np)
print('Max abs diff:', np.max(np.abs(out_ts_np - out_ckpt_np)))

# Softmax probs
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

p_ts = softmax(out_ts_np)
p_ck = softmax(out_ckpt_np)
print('TorchScript probs:', p_ts)
print('Checkpoint probs:', p_ck)
print('Max abs diff probs:', np.max(np.abs(p_ts - p_ck)))
