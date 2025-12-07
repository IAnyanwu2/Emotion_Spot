#!/usr/bin/env python3
"""Run the speech preprocessing pipeline for ABAW project.

This script validates that required Python packages and the VOSK model
are present, optionally overrides paths in the project config, and runs
the preprocessing steps that produce transcript -> punctuation ->
word_embedding -> aligned_word_embedding (`bert.npy`).

Usage example (run inside your venv):
  python3 scripts/run_speech_preprocessing.py \
    --root /path/to/raw_dataset_root \
    --output /path/to/output_root \
    --load /path/to/load_models \
    --vosk /path/to/vosk-model \
    --part -1

Notes:
 - The script imports `project.abaw5.configs.config` and mutates the
   values at runtime (no file edits required).
 - The script prints missing dependencies and exits if anything required
   is not present.
"""
import argparse
import importlib
import sys
import os


REQUIRED_PACKAGES = ['vosk', 'transformers', 'torch', 'deepmultilingualpunctuation', 'tqdm', 'pandas', 'numpy']


def check_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Raw dataset root (configs.root_directory)')
    parser.add_argument('--output', help='Output root (configs.output_root_directory)')
    parser.add_argument('--load', help='Load directory for auxiliary models (configs.load_directory)')
    parser.add_argument('--vosk', help='Path to VOSK model directory')
    parser.add_argument('--part', type=int, default=-1, help='Which part to process (-1 = all)')
    parser.add_argument('--python-package-path', default='./abaw6_preprocessing', help='Path to preprocessing package')
    args = parser.parse_args()

    missing = check_packages()
    if missing:
        print('Missing Python packages:', ', '.join(missing))
        print('Install them with pip in your venv, e.g.:')
        print('  pip install ' + ' '.join(missing))
        sys.exit(1)

    # Ensure preprocessing package is importable
    pkg_path = os.path.abspath(args.python_package_path)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

    # Import config and preprocessing class
    try:
        from project.abaw5.configs import config as project_config
        from project.abaw5.preprocessing import PreprocessingABAW5
    except Exception as e:
        print('Failed to import preprocessing package. Check --python-package-path:', e)
        sys.exit(2)

    # Mutate config values if overrides provided
    if args.root:
        project_config['root_directory'] = args.root
    if args.output:
        project_config['output_root_directory'] = args.output
    if args.load:
        project_config['load_directory'] = args.load
    if args.vosk:
        if 'speech_model' not in project_config or not isinstance(project_config['speech_model'], dict):
            project_config['speech_model'] = {'path': args.vosk}
        else:
            project_config['speech_model']['path'] = args.vosk

    # Validate VOSK model path
    vosk_path = project_config.get('speech_model', {}).get('path')
    if vosk_path is None or not os.path.isdir(vosk_path):
        print('VOSK model path not found or not a directory:', vosk_path)
        print('Please download a VOSK model and set --vosk to its folder, or set config in project.abaw5.configs')
        sys.exit(3)

    print('Configuration used:')
    print('  root_directory =', project_config['root_directory'])
    print('  output_root_directory =', project_config['output_root_directory'])
    print('  load_directory =', project_config['load_directory'])
    print('  vosk model =', project_config['speech_model']['path'])

    # Instantiate preprocessing and run the parts needed
    pre = PreprocessingABAW5(args.part, project_config)
    # Generate per-trial info and run the pipeline
    print('Generating per-trial information...')
    pre.generate_per_trial_info_dict()
    print('Running preprocessing (this may take many hours depending on data size and BERT runs)...')
    pre.prepare_data()


if __name__ == '__main__':
    main()
