#!/usr/bin/env python3
"""Repair `dataset_info.pkl` trial paths by locating existing trial folders.

This script helps when `dataset_info.pkl` contains trial paths that don't
match the on-disk layout (e.g., duplicated `processed_trials/processed_trials/...`).

Usage:
  python3 scripts/repair_dataset_info.py --pkl /opt/dataset/dataset_info.pkl \
      --processed-root /opt/dataset/processed_trials --out-pkl /opt/dataset/dataset_info.fixed.pkl

The script will try to resolve each `trial` entry to an existing directory
under `processed_root`. If it finds a match it replaces the trial entry with
the relative path under `processed_root`. It writes a repaired pickle and
prints a short summary.
"""
import argparse
import os
import pickle
from pathlib import Path


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_pkl(p, data):
    with open(p, 'wb') as f:
        pickle.dump(data, f)


def find_best_match(processed_root: Path, trial: str):
    # First try exact path under processed_root
    candidate = processed_root / trial
    if candidate.is_dir():
        return trial

    # Try the basename (last segment)
    base = Path(trial).name
    matches = list(processed_root.rglob(base))
    for m in matches:
        if m.is_dir():
            return str(m.relative_to(processed_root))

    # Try matching by removing potential duplicated prefix like 'processed_trials/'
    parts = Path(trial).parts
    for i in range(1, len(parts)):
        sub = Path(*parts[i:])
        cand = processed_root / sub
        if cand.is_dir():
            return str(sub)

    # No match found
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', required=True, help='Input dataset_info.pkl')
    p.add_argument('--processed-root', default='/opt/dataset/processed_trials', help='Processed trials root')
    p.add_argument('--out-pkl', default=None, help='Path to write repaired dataset_info.pkl (defaults to overwrite input)')
    args = p.parse_args()

    inp = Path(args.pkl)
    if not inp.is_file():
        raise SystemExit(f"Input pickle not found: {inp}")

    processed_root = Path(args.processed_root)
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")

    dataset_info = load_pkl(str(inp))
    trials = dataset_info.get('trial', [])

    repaired = []
    missing = []
    fixes = 0

    for t in trials:
        best = find_best_match(processed_root, t)
        if best is None:
            missing.append(t)
            repaired.append(t)
        else:
            if best != t:
                fixes += 1
            repaired.append(best)

    dataset_info['trial'] = repaired

    out = Path(args.out_pkl) if args.out_pkl else inp
    save_pkl(str(out), dataset_info)

    print(f"Wrote repaired dataset_info to: {out}")
    print(f"Trials total: {len(trials)}; fixed: {fixes}; unresolved: {len(missing)}")
    if missing:
        print("Some trials could not be resolved (first 10 shown):")
        for m in missing[:10]:
            print(" -", m)


if __name__ == '__main__':
    main()
