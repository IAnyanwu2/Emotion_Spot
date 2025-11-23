# scripts/transcribe_all.py
import whisper
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--wav-dir", required=True)
parser.add_argument("--model", default="tiny", help="whisper model: tiny, base, small, medium, large")
parser.add_argument("--out-ext", default=".txt")
args = parser.parse_args()

model = whisper.load_model(args.model)

wav_dir = Path(args.wav_dir)
wav_files = list(wav_dir.rglob("*.wav"))
print(f"Found {len(wav_files)} wav files")

for i, wav in enumerate(wav_files, 1):
    out_txt = wav.with_suffix(args.out_ext)
    if out_txt.exists():
        print(f"[{i}/{len(wav_files)}] Skipping (exists) {wav.name}")
        continue
    print(f"[{i}/{len(wav_files)}] Transcribing {wav.name} ...")
    result = model.transcribe(str(wav))
    text = result["text"].strip()
    out_txt.write_text(text, encoding="utf-8")
print("Done")