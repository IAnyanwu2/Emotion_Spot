# Inference & Results — Summary

This document focuses on inference workflows, diagrams, and results collected while testing the two emotion models in this repo.

## High-level Architecture (ASCII diagram)

```
Data Sources
  ├─ FER2013 (images)
  ├─ RAVDESS Audio (wav)
  └─ RAVDESS Video (mp4)
       ↓
Training (massive multimodal model)
  - train_multimodal_massive.py (CNN + LSTM) -> MultimodalEmotionCNN
  - Input: (B, T, C, H, W), T=10, ImageNet-normalized
       ↓
Export -> TorchScript (models/massive_emotion_model_ts.pt)
       ↓
Inference pipeline (scripts/infer_on_spot.py)
  - ffmpeg -> extract frames & audio
  - Preprocess frames (resize 48x48, normalize, channel order fix)
  - Compute optional audio embedding (torchaudio)
  - Model inference -> save CSV with per-frame predictions & probabilities

Edge (Spot) model path
  - train_spot_emotion.py -> SimpleEmotionCNN (lightweight CNN + LSTM)
  - Input: expected: (B, T, H, W, C) in training pipeline; no ImageNet normalization
  - Export -> TorchScript (models/spot_emotion_model_6class_ts.pt)
  - Inference via scripts/infer_on_spot.py (use `--no-normalize`)
```

---

## Key Workflows

1) Export to TorchScript
```
python -c "import torch; from train_multimodal_massive import MultimodalEmotionCNN; sd=torch.load('massive_emotion_model.pth', map_location='cpu'); m=MultimodalEmotionCNN(num_classes=6,input_size=48); m.load_state_dict(sd); m.eval(); ts=torch.jit.script(m); ts.save('models/massive_emotion_model_ts.pt')"
```
Note: ensure export step respects input order (channels-first for `massive` model) and dtype.

2) Local inference (massive):
```
python .\scripts\infer_on_spot.py --model .\models\massive_emotion_model_ts.pt --video "<PATH_TO_MP4>" --out massive_preds.csv --device cpu
```
- Default behavior: uses ImageNet normalization and T=10 frames (sampling/trim/repeat)

3) Local inference (Spot model):
```
python .\scripts\infer_on_spot.py --model .\models\spot_emotion_model_6class_ts.pt --video "<PATH_TO_MP4>" --out spot_preds.csv --device cpu --no-normalize
```
- `--no-normalize` flag is required for the Spot model because `train_spot_emotion.py` uses raw [0,1] pixel values (no ImageNet normalization)
- If faces are off-grid for slow robots or far cameras, add face cropping before inference (see `Face-crop` section below).

4) Debugging inference
- Use `--debug` in `infer_on_spot.py` to write sampled frames and `logits.npy` / `probs.npy` and a `summary.txt` next to output CSV. Attach the debug folder to get help diagnosing.
- Parity check: use `tools/test_model_on_image.py` to compare checkpoint vs TorchScript predictions for the same input.

## Reproducibility & Parity Tests
- Clone the repo and use the `tools/test_model_on_image.py` script to compare raw checkpoint vs TorchScript model outputs. If parity is off, re-export with `torch.jit.trace` or re-check model forward for non-deterministic ops.

- The `--debug` option writes sampled frames to `<out_stem>_debug` and saves `logits.npy` & `probs.npy` which are useful when reporting to others.

---

## Next Steps (Recommended)
1. Run `tools/test_model_on_image.py` for both models to check parity.
2. Enable `--face-crop` if many frames show off-face predictions.
3. If parity is OK and face-crop doesn't change outcomes, re-evaluate dataset balance and consider fine-tuning with labeled data to correct bias.
4. Add typical model card and update model README with `Label mapping`, `Normalization assumptions`, `Input shape`, `Expected runtime`, and `Accuracy`.
5. Push to your GitHub repo using the `scripts/push_to_github.ps1` helper. This script will:
  - Add the repo to Git Safe Directory for Windows
  - Enable Git LFS and track *.pt and *.pth
  - Commit the staged changes and push to `inference-docs`
  - Note: after pushing, create a Pull Request (PR) for review and merge to `main`.

---

## Where to get help
- Attach the `_debug` folder (from `--debug`) and CSV when opening an issue; include the `summary.txt` and a few sample frames (frame_000xxx.jpg) so reviewers can quickly see input vs logits.

---

### Quick Links to Important Files
- `scripts/infer_on_spot.py` — inference logic and CLI flags
- `scripts/export_to_torchscript.py` — export tooling
- `train_multimodal_massive.py` — massive model architecture & dataset
- `train_spot_emotion.py` — Spot model & dataset
- `tools/test_model_on_image.py` — Added parity test script (see repo `tools/`)
