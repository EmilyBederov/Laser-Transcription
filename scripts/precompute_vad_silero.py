"""
Precompute Silero-VAD segments for each audio file in a CSV.

Each sidecar is a [K, 2] int32 array: K merged (start, end) sample indices
defining the speech-only regions of the audio, AFTER:
  1. Silero get_speech_timestamps(threshold, min_speech_ms)
  2. Expand each segment by ±expand_ms
  3. Merge segments whose gap < merge_gap_ms

At re-cache / training time, both teacher (audio) and student (laser/laser)
crop their waveforms identically using these segments before computing mel.

Idempotent — already-existing sidecar files are skipped.

Usage:
    python scripts/precompute_vad_silero.py \\
        --csv data/train.csv \\
        --out_dir /path/to/vad_sidecar_silero/train \\
        --threshold 0.5 --min_speech_ms 200 --expand_ms 200 --merge_gap_ms 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

SR = 16000


def load_mono_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)  # [T]


def expand_and_merge(segments, expand_samples, merge_gap_samples, n_total):
    """
    segments: list of {'start': int, 'end': int} from Silero.
    Returns: list of (start, end) tuples after expand + merge, clipped to [0, n_total].
    """
    if not segments:
        return []

    expanded = []
    for s in segments:
        start = max(0, int(s["start"]) - expand_samples)
        end = min(n_total, int(s["end"]) + expand_samples)
        expanded.append((start, end))

    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for start, end in expanded[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap < merge_gap_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--audio_col", default="audio_path")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_speech_ms", type=int, default=200)
    p.add_argument("--expand_ms", type=int, default=200)
    p.add_argument("--merge_gap_ms", type=int, default=300)
    p.add_argument("--force", action="store_true", help="Overwrite existing sidecar files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata once per directory
    meta_path = out_dir / "_meta.json"
    meta = {
        "vad": "silero",
        "threshold": args.threshold,
        "min_speech_ms": args.min_speech_ms,
        "expand_ms": args.expand_ms,
        "merge_gap_ms": args.merge_gap_ms,
        "sample_rate": SR,
        "csv": args.csv,
    }
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Wrote metadata -> {meta_path}")
    else:
        existing_meta = json.loads(meta_path.read_text())
        for k in ("threshold", "min_speech_ms", "expand_ms", "merge_gap_ms"):
            if existing_meta.get(k) != meta[k]:
                print(f"[WARN] meta mismatch on {k}: existing={existing_meta.get(k)} new={meta[k]}")

    expand_samples = int(args.expand_ms * SR / 1000)
    merge_gap_samples = int(args.merge_gap_ms * SR / 1000)

    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    model.eval()
    get_speech_ts = utils[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Silero on {device}\n")

    df = pd.read_csv(args.csv).dropna(subset=[args.audio_col]).reset_index(drop=True)
    print(f"{len(df)} samples in {args.csv}")

    n_written = n_skipped = n_failed = n_empty = 0
    speech_durations = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Silero VAD"):
        out_file = out_dir / f"sample_{idx:06d}.npy"
        if out_file.exists() and not args.force:
            n_skipped += 1
            continue

        try:
            wav = load_mono_16k(row[args.audio_col])
            n_total = wav.shape[0]
            wav = wav.to(device)

            with torch.no_grad():
                raw_segments = get_speech_ts(
                    wav, model,
                    threshold=args.threshold,
                    min_speech_duration_ms=args.min_speech_ms,
                    sampling_rate=SR,
                )

            merged = expand_and_merge(raw_segments, expand_samples, merge_gap_samples, n_total)

            if not merged:
                arr = np.zeros((0, 2), dtype=np.int32)
                n_empty += 1
            else:
                arr = np.asarray(merged, dtype=np.int32)
                speech_dur = (arr[:, 1] - arr[:, 0]).sum() / SR
                speech_durations.append(speech_dur)

            np.save(out_file, arr)
            n_written += 1
        except Exception as e:
            print(f"  [WARN] idx={idx} audio={row[args.audio_col]}: {e}")
            n_failed += 1

    mean_speech = float(np.mean(speech_durations)) if speech_durations else 0.0
    print(f"\nDone. written={n_written} skipped={n_skipped} empty={n_empty} failed={n_failed}")
    print(f"Mean speech duration (new samples): {mean_speech:.2f}s")
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
