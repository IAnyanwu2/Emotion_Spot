#!/usr/bin/env python3
"""Export a trained checkpoint to a TorchScript file for deployment.

Usage patterns:
- If your checkpoint is already TorchScript: it will be copied to the output path.
- If your checkpoint is a saved full model object (torch.save(model)), it will be scripted and saved.
- If your checkpoint contains only a `state_dict`, you must pass `--model-class` and any required constructor args
  (for example, `--backbone-state-dict` and `--root-dir`) so the script can instantiate the correct model and
  load the state_dict before scripting.

This script supports the model classes defined in `models/model.py` (e.g. `RCMA`, `LeaderFollowerAttentionNetwork`).
It attempts to be conservative and will error with guidance if it cannot proceed automatically.
"""

import argparse
import inspect
import os
import shutil
import sys
import torch


def is_torchscript(path):
    try:
        _ = torch.jit.load(path, map_location='cpu')
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to checkpoint (.pth/.pt) or TorchScript file')
    p.add_argument('--out', required=True, help='Output TorchScript path (.pt)')
    p.add_argument('--device', default='cpu', help='Device to load on for scripting (cpu/cuda)')
    p.add_argument('--model-class', default=None, help='Model class name to instantiate if checkpoint is state_dict (e.g. RCMA, LeaderFollowerAttentionNetwork)')
    p.add_argument('--backbone-state-dict', default=None, help='Backbone state dict name (without .pth) if needed by the model constructor')
    p.add_argument('--root-dir', default='.', help='Root dir used by model when loading backbone files')
    p.add_argument('--modalities', default=None, help='Comma-separated modalities list, e.g. video,logmel')
    args = p.parse_args()

    ckpt_path = args.ckpt
    out_path = args.out
    device = args.device

    # If file is already TorchScript, copy to out
    if is_torchscript(ckpt_path):
        print(f"Input appears to be TorchScript. Copying {ckpt_path} -> {out_path}")
        shutil.copyfile(ckpt_path, out_path)
        print('Done')
        return

    # Not TorchScript: try torch.load
    loaded = torch.load(ckpt_path, map_location='cpu')

    # If loaded is a Module
    if isinstance(loaded, torch.nn.Module):
        model = loaded
        model.to(device)
        model.eval()
        print('Scripting full model object...')
        scripted = torch.jit.script(model)
        scripted.save(out_path)
        print('Saved TorchScript to', out_path)
        return

    # If loaded is a dict
    if isinstance(loaded, dict):
        # If it contains a 'model' key with a Module
        if 'model' in loaded and isinstance(loaded['model'], torch.nn.Module):
            model = loaded['model']
            model.to(device)
            model.eval()
            print("Scripting model from checkpoint['model']...")
            scripted = torch.jit.script(model)
            scripted.save(out_path)
            print('Saved TorchScript to', out_path)
            return

        # If it contains only state_dict
        if 'state_dict' in loaded or all(isinstance(k, str) for k in loaded.keys()):
            state_dict = loaded.get('state_dict', loaded)
            if args.model_class is None:
                raise RuntimeError('Checkpoint looks like a state_dict. Please provide --model-class to construct the model before loading state_dict.')

            # Support module-qualified model class names like 'train_spot_emotion.SimpleEmotionCNN'
            sys.path.insert(0, os.getcwd())
            if '.' in args.model_class:
                mod_name, class_name = args.model_class.rsplit('.', 1)
                try:
                    mod = __import__(mod_name, fromlist=[class_name])
                except Exception as e:
                    raise RuntimeError(f"Failed to import module '{mod_name}': {e}")
                if not hasattr(mod, class_name):
                    raise RuntimeError(f"Module '{mod_name}' does not define class '{class_name}'")
                ModelClass = getattr(mod, class_name)
            else:
                # Try models.model first, then fall back to top-level modules
                try:
                    from models import model as model_module
                    if hasattr(model_module, args.model_class):
                        ModelClass = getattr(model_module, args.model_class)
                    else:
                        # search top-level modules
                        mod = __import__(args.model_class.lower(), fromlist=[args.model_class])
                        if hasattr(mod, args.model_class):
                            ModelClass = getattr(mod, args.model_class)
                        else:
                            raise RuntimeError()
                except Exception:
                    raise RuntimeError(f"Model class '{args.model_class}' not found. Try using module.Class format, e.g. 'train_spot_emotion.SimpleEmotionCNN'.")

            modalities = None
            if args.modalities:
                modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]

            # Try to instantiate with common constructor signature
            kwargs = {}
            # Many model constructors expect backbone_state_dict and modality
            if args.backbone_state_dict:
                kwargs['backbone_state_dict'] = args.backbone_state_dict
            if modalities is not None:
                kwargs['modality'] = modalities
            # Also include num_classes/input_size if user passed via modal options (not required here)
            # We'll attempt to only pass parameters that the constructor accepts to avoid unexpected-arg errors

            # Filter kwargs to only those accepted by the constructor
            try:
                sig = inspect.signature(ModelClass.__init__)
                accepted = set(sig.parameters.keys()) - {'self', '*', '**', 'args', 'kwargs'}
            except Exception:
                accepted = set(kwargs.keys())

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            dropped = {k: v for k, v in kwargs.items() if k not in accepted}
            if dropped:
                print('Note: dropping unsupported constructor args:', list(dropped.keys()))

            print('Instantiating model', args.model_class, 'with args', filtered_kwargs)
            model = ModelClass(**filtered_kwargs)

            # Some classes require calling init()
            if hasattr(model, 'init'):
                try:
                    model.init()
                except Exception as e:
                    print('Warning: model.init() raised', e)

            # Load state dict
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                # try to be helpful
                raise RuntimeError(f'Failed to load state_dict into model: {e}')

            model.to(device)
            model.eval()
            print('Scripting constructed model...')
            scripted = torch.jit.script(model)
            scripted.save(out_path)
            print('Saved TorchScript to', out_path)
            return

    raise RuntimeError('Unrecognized checkpoint format. Provide a TorchScript file, a saved model object, or a state_dict with --model-class.')


if __name__ == '__main__':
    main()
