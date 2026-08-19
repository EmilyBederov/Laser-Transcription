# Direct Speech Transcription from Laser Doppler Vibrometry

Code for transcribing speech directly from laser Doppler vibrometry (LDV)
measurements, without an intermediate reconstructed waveform. A student encoder
reads the laser/LDV signal and emits representations that a frozen pretrained
Whisper decoder can transcribe.

## How it works

The LDV signal is turned into an 80-bin mel spectrogram and fed to a **Whisper
small.en encoder student**. That student is trained so its output is readable by
a **frozen Whisper decoder**, in two stages:

1. **Stage 1 — decoder-guided pretraining.** The Whisper decoder is frozen and
   used as a task oracle: cross-entropy gradients flow through it into the
   encoder only. This is combined with multi-layer feature distillation from the
   Whisper *audio* encoder (MSE + cosine on selected layers, plus relational KD
   on the deep blocks). The decoder is never adapted here.
2. **Stage 2 — joint adaptation.** The encoder is fine-tuned in full (layer-wise
   LR decay) while the decoder is adapted with low-rank (LoRA) updates on its
   self- and cross-attention projections. A KL term keeps the decoder attending
   to the same encoder positions the audio teacher would.

The synthetic and real corpora are trained as **separate tracks** (Stage 1 then
Stage 2 within each track), never mixed within a stage.

## Repository layout

```
src/
  models/       Whisper small.en encoder student (whisper_student.py), KD system
  lightning/    training module (system.py), data modules, cached dataset
  loss/         composite distillation loss (CE, multi-layer MSE/cosine, RKD)
  utils/        EMA, transcription helpers
  train.py      training entry point (Hydra)
scripts/
  eval.py                  WER/CER evaluation (beam search, no-repeat-ngram, optional VAD)
  precompute_vad_silero.py Silero VAD sidecars for offline training-time cropping (optional)
configs/
  config.yaml              top-level defaults (model + data + trainer)
  model/                   whisper_small_student.yaml
  data/                    default.yaml (CSV paths, mel settings, cache dir)
  trainer/                 decoder_guided_encoder_only.yaml (S1), stage2_xattn.yaml (S2)
```

## Installation

```bash
conda create -n ldv python=3.10 && conda activate ldv
pip install -r requirements.txt
```

Silero VAD and the Whisper weights are downloaded automatically at first use
(via `torch.hub` / the `openai-whisper` package). A CUDA GPU is required for
training; evaluation runs on a single GPU.

### 1. Data and CSV manifests

The corpora are released separately as **LaserSpeech** (synthesis pipeline,
synthetic corpus, and real LDV recordings):
<https://github.com/EmilyBederov/LaserSpeech>.

Prepare a CSV manifest for each split with three columns:

```csv
laser_path,audio_path,text
/data/ldv/utt001.wav,/data/clean/utt001.wav,"the quick brown fox"
```

- `laser_path` — the LDV / laser recording (student input).
- `audio_path` — the aligned clean-speech reference (used only to compute the
  teacher features that supervise Stage 1; not needed at inference).
- `text` — the ground-truth transcript.

Point the configs at your CSVs (edit `configs/data/default.yaml` or override on
the command line):

```yaml
train_csv: "path/to/your/train.csv"
val_csv:   "path/to/your/val.csv"
cache_dir: "path/to/your/cache"   # teacher-feature cache is written here
lowercase_labels: false             # true for synthetic/LibriSpeech, false for laser
```

### 2. The teacher-feature cache (built automatically)

With `data.use_cached=true` (the default), the **first training run precomputes
the Whisper teacher features and writes them to `cache_dir`**, one file per
sample. This first pass is slow and disk-heavy; every run afterward loads the
cache and is much faster. You do not run a separate build step — just make sure
`cache_dir` points at a location with enough free space.

## Training

Both configs default to **3 GPUs** (`devices: 3, strategy: ddp`). For a single
GPU, add `trainer.devices=1 trainer.strategy=auto` to every command.

### Stage 1 — decoder-guided pretraining

```bash
python -m src.train \
    model=whisper_small_student \
    trainer=decoder_guided_encoder_only \
    data.train_csv=path/to/your/train.csv \
    data.val_csv=path/to/your/val.csv \
    data.use_cached=true
```

Key Stage-1 knobs live in `configs/trainer/decoder_guided_encoder_only.yaml`:
`w_task=1.0` is the decoder-guidance term (set it to `0` to recover the
feature-only control), `w_feat=0.5` is the feature anchor, and the multi-layer
distillation layers are set in `configs/model/whisper_small_student.yaml`.

### Stage 2 — joint fine-tune + decoder LoRA

Feed Stage 2 the encoder from your best Stage-1 checkpoint via
`trainer.encoder_ckpt_path`:

```bash
python -m src.train \
    model=whisper_small_student \
    trainer=stage2_xattn \
    data.train_csv=path/to/your/train.csv \
    data.val_csv=path/to/your/val.csv \
    trainer.encoder_ckpt_path=outputs/<stage1-run>/checkpoints/last.ckpt
```

Checkpoints and Hydra configs for each run land under
`outputs/<exp_name>/`. Stage 2 checkpoints on `val/wer`; Stage 1 on `val/loss`.

## Evaluation

At inference, silence is cropped from the **laser signal** with Silero VAD
(`--vad`), matching the training crop; the VAD runs on the LDV recording itself,
never on a clean reference.

```bash
python scripts/eval.py \
    --checkpoint outputs/<stage2-run>/checkpoints/best.ckpt \
    --test_csv path/to/your/test.csv \
    --beam_size 5 --no_repeat_ngram_size 3 --vad \
    --output results/eval.csv
```

The script prints WER/CER and writes per-utterance predictions to `--output`.

| flag | meaning |
|---|---|
| `--vad` | Silero-VAD crop the laser signal (recommended for real LDV). |
| `--beam_size 5` | Beam search (1 = greedy). |
| `--no_repeat_ngram_size 3` | Block repeated n-grams (suppresses decoder loops). |

