"""
Full evaluation script: WER + CER on a test CSV.

Usage:
    python scripts/eval.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --test_csv data/laser_b12b13b14_xy_val.csv \
        --output results/eval.csv \
        --beam_size 5

Loads the full Lightning module (encoder + LoRA decoder + EMA), runs real
autoregressive decoding on every sample, computes WER/CER/MER/WIL.
"""

import sys
import os
import argparse
import re
import torch
import torch.nn.functional as F
import whisper
from jiwer import wer, cer, mer, wil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# PyTorch 2.6+ fix
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import torchaudio
from src.lightning.system import RadarLightningModule
from src.lightning.datamodule import (
    load_and_pad_audio, get_radar_mel_spectrogram,
    get_radar_linear_spectrogram, SAMPLE_RATE, N_SAMPLES,
)

_NORMALIZER = EnglishTextNormalizer()


# ---------------------------------------------------------------------------
# Silero VAD applied to the LASER signal at inference: crop voiced segments and
# pad to 30 s, with the SAME parameters the training cache used
# (scripts/precompute_vad_silero.py). This aligns inference with training and
# removes the silent stretches Whisper tends to hallucinate on. VAD runs on the
# laser signal itself (no clean reference), i.e. the deployable pipeline.
# ---------------------------------------------------------------------------
def load_silero_vad():
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                  model="silero_vad", force_reload=False, onnx=False)
    model.eval()
    return model, utils[0]          # (model, get_speech_timestamps)


def _expand_and_merge(segments, expand_samples, merge_gap_samples, n_total):
    if not segments:
        return []
    exp = [(max(0, int(s["start"]) - expand_samples),
            min(n_total, int(s["end"]) + expand_samples)) for s in segments]
    exp.sort(key=lambda x: x[0])
    merged = [exp[0]]
    for st, en in exp[1:]:
        ps, pe = merged[-1]
        if st - pe < merge_gap_samples:
            merged[-1] = (ps, max(pe, en))
        else:
            merged.append((st, en))
    return merged


def vad_crop_laser(path, silero_model, get_speech_timestamps, device,
                   threshold=0.5, min_speech_ms=200, expand_ms=200, merge_gap_ms=300):
    """Load laser wav, Silero-VAD crop voiced segments, pad/trim to 30 s -> [1, N_SAMPLES]."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.squeeze(0)            # [T]
    n_total = wav.shape[0]
    with torch.no_grad():
        ts = get_speech_timestamps(wav.to(device), silero_model,
                                   threshold=threshold,
                                   min_speech_duration_ms=min_speech_ms,
                                   sampling_rate=SAMPLE_RATE)
    segs = _expand_and_merge(ts, expand_ms * SAMPLE_RATE // 1000,
                             merge_gap_ms * SAMPLE_RATE // 1000, n_total)
    if segs:
        chunks = [wav[s:e] for s, e in segs if e > s]
        cropped = torch.cat(chunks) if chunks else wav.new_zeros(0)
    else:
        cropped = wav.new_zeros(0)
    out = wav.new_zeros(N_SAMPLES)
    L = min(cropped.shape[0], N_SAMPLES)
    out[:L] = cropped[:L]
    return out.unsqueeze(0)          # [1, N_SAMPLES]


def strip_special_tokens(text):
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return ' '.join(text.split()).strip()


def _get_sot_sequence(tokenizer, legacy_sot=False):
    """Return the SOT prefix for decoding.

    Default: Whisper's canonical sot_sequence_including_notimestamps. For small.en
        this is [sot, notimestamps] = [50257, 50362].
    legacy_sot=True: the buggy 3-token prefix [sot, transcribe, notimestamps]
        used by old training/eval. Pass this when evaluating ckpts that were
        trained with this prefix (everything before the SOT fix).
    """
    if legacy_sot:
        return [
            tokenizer.sot,
            tokenizer.special_tokens["<|transcribe|>"],
            tokenizer.special_tokens["<|notimestamps|>"],
        ]
    return list(tokenizer.sot_sequence_including_notimestamps)


def greedy_decode(decoder, embeddings, tokenizer, max_len=445, legacy_sot=False):
    """Autoregressive greedy decode. max_len=445 = Whisper max (448) minus 3-token prefix."""
    device = embeddings.device
    batch_size = embeddings.shape[0]
    sot_sequence = _get_sot_sequence(tokenizer, legacy_sot)
    tokens = torch.tensor([sot_sequence] * batch_size, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = decoder(tokens, embeddings)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        if (next_token == tokenizer.eot).all():
            break
    results = []
    for t in tokens:
        results.append(strip_special_tokens(tokenizer.decode(t.tolist())))
    return results


def _banned_ngram_tokens(seq, n, prompt_len):
    """Tokens that would complete a repeated n-gram if appended to seq.

    Blocks degenerate repetition loops (e.g. "penny penny penny ...") the way
    HuggingFace's no_repeat_ngram_size does: for the current (n-1)-token suffix,
    forbid any next token that has already followed that same suffix earlier in
    the generated portion of the sequence. prompt_len excludes the SOT prefix so
    the prompt tokens are not treated as generated context.
    """
    if n <= 0 or len(seq) < n:
        return set()
    gen = seq[prompt_len:]
    if len(gen) < n:
        return set()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = set()
    # scan the generated portion for the same (n-1)-prefix
    for i in range(len(gen) - n + 1):
        if tuple(gen[i:i + n - 1]) == prefix:
            banned.add(gen[i + n - 1])
    return banned


def beam_decode(decoder, embeddings, tokenizer, beam_size=5, max_len=445,
                legacy_sot=False, no_repeat_ngram_size=3):
    """Beam search decode — runs one sample at a time (batch_size=1 only).

    no_repeat_ngram_size blocks repeated n-grams (default 3); set 0 to disable.
    """
    device = embeddings.device
    sot_sequence = _get_sot_sequence(tokenizer, legacy_sot)
    prompt_len = len(sot_sequence)
    # Each beam: (log_prob, token_list)
    beams = [(0.0, sot_sequence[:])]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        for log_prob, seq in beams:
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            logits = decoder(tokens, embeddings)           # [1, len, vocab]
            log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
            banned = _banned_ngram_tokens(seq, no_repeat_ngram_size, prompt_len)
            if banned:
                idx = torch.tensor(list(banned), device=log_probs.device)
                log_probs = log_probs.index_fill(0, idx, float('-inf'))
            top_log_probs, top_ids = log_probs.topk(beam_size)
            for lp, tid in zip(top_log_probs.tolist(), top_ids.tolist()):
                all_candidates.append((log_prob + lp, seq + [tid]))

        # Keep top beam_size, move completed beams out
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for log_prob, seq in all_candidates[:beam_size * 2]:
            if seq[-1] == tokenizer.eot:
                completed.append((log_prob, seq))
            else:
                beams.append((log_prob, seq))
            if len(beams) == beam_size:
                break

        if len(completed) >= beam_size or not beams:
            break

    if not completed:
        completed = beams
    best_seq = max(completed, key=lambda x: x[0])[1]
    return strip_special_tokens(tokenizer.decode(best_seq))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate radar-to-speech model')
    parser.add_argument('--checkpoint', required=True, help='Path to .ckpt file')
    parser.add_argument('--test_csv', required=True, help='CSV with radar_path and text columns')
    parser.add_argument('--output', default='eval_results.csv', help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for eval')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size (1=greedy, 5 recommended)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3,
                        help='Block repeated n-grams (default 3; 0 disables). Kills repetition loops.')
    parser.add_argument('--vad', action='store_true',
                        help='Silero-VAD crop the laser signal (voiced segments, pad to 30s) '
                             'before the mel, matching the training crop. Runs on the laser '
                             'itself (no clean reference); removes silence Whisper hallucinates on.')
    parser.add_argument('--legacy_sot', action='store_true',
                        help='Use the buggy 3-token SOT [sot, transcribe, notimestamps]. '
                             'Set this for ckpts trained BEFORE the SOT fix '
                             '(S1_WSMALL_V2, S2_RESUME_FIXED, S2_LASER_LORA_70191, T17, T25, T27, T31, etc).')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load full Lightning module
    print(f'Loading checkpoint: {args.checkpoint}')
    module = RadarLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.eval()
    module.to(device)

    student = module.ema.ema_model if module.use_ema else module.system.student
    student = student.to(device)
    decoder = module.system.teacher.decoder.to(device)
    tokenizer = whisper.tokenizer.get_tokenizer(False, task='transcribe')

    cfg = module.cfg
    use_mel = cfg.data.get('use_mel_radar', False)
    n_mel = cfg.data.get('n_mel_radar', 80)
    mel_fmax = cfg.data.get('mel_fmax', 8000)
    mel_n_fft = cfg.data.get('mel_n_fft', 400)
    lpf_cutoff_hz = cfg.data.get('lpf_cutoff_hz', None)
    normalize = cfg.data.get('normalize_radar_spec', False)

    print(f'Preprocessing: {"mel " + str(n_mel) + " bins fmax=" + str(mel_fmax) if use_mel else "linear lpf=" + str(lpf_cutoff_hz)}')

    # Silero VAD (laser-side crop) if requested
    silero = None
    if args.vad:
        print('Loading Silero VAD for laser-side cropping...')
        _svm, _gst = load_silero_vad()
        silero = (_svm.to(device), _gst)

    # Load test CSV
    df = pd.read_csv(args.test_csv).dropna(subset=['text'])
    print(f'Evaluating {len(df)} samples...  (VAD crop: {bool(args.vad)})\n')

    all_gts, all_preds = [], []
    errors = 0

    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                if args.vad:
                    wav = vad_crop_laser(row['radar_path'], silero[0], silero[1], device)
                else:
                    wav, _ = load_and_pad_audio(row['radar_path'])
                if use_mel:
                    spec = get_radar_mel_spectrogram(wav, n_mels=n_mel, fmax=mel_fmax,
                                                      n_fft=mel_n_fft, normalize=normalize)
                else:
                    spec = get_radar_linear_spectrogram(wav, lpf_cutoff_hz=lpf_cutoff_hz,
                                                         normalize=normalize)
                spec = spec.unsqueeze(0).to(device)  # [1, 1, freq, 3000]

                feats = student(spec)
                if isinstance(feats, tuple):
                    feats = feats[0]
                feats = F.layer_norm(feats.float(), [feats.shape[-1]]).to(feats.dtype)
                if args.beam_size > 1:
                    pred = beam_decode(decoder, feats, tokenizer,
                                       beam_size=args.beam_size,
                                       legacy_sot=args.legacy_sot,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size)
                else:
                    pred = greedy_decode(decoder, feats, tokenizer,
                                         legacy_sot=args.legacy_sot)[0]

                gt = str(row['text']).strip()
                all_gts.append(gt)
                all_preds.append(pred)

                if i < 5:
                    print(f'[{i}] GT:   {gt}')
                    print(f'[{i}] PRED: {pred}')
                    print()

            except Exception as e:
                print(f'Error on sample {i}: {e}')
                all_gts.append(str(row['text']).strip())
                all_preds.append('')
                errors += 1

    # Metrics — WhisperNormalizer handles lowercase, punctuation, contractions, numbers
    norm_gts   = [_NORMALIZER(g) for g in all_gts]
    norm_preds = [_NORMALIZER(p) for p in all_preds]

    final_wer = wer(norm_gts, norm_preds)
    final_cer = cer(norm_gts, norm_preds)
    final_mer = mer(norm_gts, norm_preds)
    final_wil = wil(norm_gts, norm_preds)

    print(f'\n{"="*60}')
    print(f'EVALUATION RESULTS  ({len(all_gts)} samples, {errors} errors)')
    print(f'{"="*60}')
    print(f'WER:  {final_wer * 100:.2f}%')
    print(f'CER:  {final_cer * 100:.2f}%')
    print(f'MER:  {final_mer * 100:.2f}%')
    print(f'WIL:  {final_wil * 100:.2f}%')
    print(f'{"="*60}\n')

    # Save CSV
    out_df = df.copy().reset_index(drop=True)
    out_df['prediction'] = all_preds
    out_df['wer'] = [
        wer(_NORMALIZER(g), _NORMALIZER(p))
        for g, p in zip(all_gts, all_preds)
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f'Saved results → {args.output}')

    # Save metrics txt
    metrics_path = args.output.replace('.csv', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Test CSV:   {args.test_csv}\n')
        f.write(f'Samples:    {len(all_gts)} ({errors} errors)\n\n')
        f.write(f'WER:  {final_wer * 100:.2f}%\n')
        f.write(f'CER:  {final_cer * 100:.2f}%\n')
        f.write(f'MER:  {final_mer * 100:.2f}%\n')
        f.write(f'WIL:  {final_wil * 100:.2f}%\n')
    print(f'Saved metrics → {metrics_path}')

    return final_wer


if __name__ == '__main__':
    main()
