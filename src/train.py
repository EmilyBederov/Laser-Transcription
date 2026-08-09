"""
Main training script using PyTorch Lightning + Hydra
"""

import os
import re
import glob
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
import pandas as pd
from torch.utils.data import DataLoader
import whisper

# Fix for PyTorch 2.6+ checkpoint loading - disable weights_only for our own checkpoints
# Monkey-patch torch.load to force weights_only=False (safe for our own checkpoints)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Force disable - PyTorch Lightning passes True explicitly
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from src.lightning.datamodule import RADARdataset
from src.lightning.data_cached import RADARdatasetCached
from src.lightning.system import RadarLightningModule
from src.lightning.callbacks import LogPredictionsCallback, LogAttnMapsCallback


def create_dataloaders(cfg):
    """Create train and validation dataloaders"""
    print(f"Loading data from: {cfg.data.train_csv} and {cfg.data.val_csv}")
    
    # Load CSVs
    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)

    # Filter out rows with NaN text
    train_df = train_df.dropna(subset=['text'])
    val_df = val_df.dropna(subset=['text'])

    # Optionally lowercase labels (use for synth/LibriSpeech data which is ALL CAPS)
    if cfg.data.get('lowercase_labels', False):
        train_df['text'] = train_df['text'].str.lower()
        val_df['text'] = val_df['text'].str.lower()
        print(">> Labels lowercased (lowercase_labels=true)")

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    
    # --- DETECT MEL BINS ---
    # Large-v3 requires 128 mels, others use 80
    if "large-v3" in cfg.model.teacher_model:
        n_mels = 128
        print(">> Detected Whisper Large-v3: Using 128 Mel Bands")
    else:
        n_mels = 80
        print(">> Using Standard 80 Mel Bands")

    # --- Choose Dataset Type ---
    use_cached = cfg.data.get('use_cached', False)

    if use_cached:
        print(">> Using CACHED dataset (fast loading)")
        # Create tokenizer ONCE in main process (avoid DataLoader worker deadlock)
        tokenizer = whisper.tokenizer.get_tokenizer(False, task="transcribe")  # English-only

        use_mel_radar = cfg.data.get('use_mel_radar', False)
        n_mel_radar = cfg.data.get('n_mel_radar', 80)
        mel_fmax = cfg.data.get('mel_fmax', 2500)
        mel_n_fft = cfg.data.get('mel_n_fft', 1024)
        dual_mel_radar = cfg.data.get('dual_mel_radar', False)

        train_dataset = RADARdatasetCached(
            radar_paths=train_df['radar_path'].tolist(),
            audio_paths=train_df['audio_path'].tolist(),
            texts=train_df['text'].tolist(),
            tokenizer=tokenizer,
            device='cpu',
            train=True,
            cache_dir=cfg.data.get('cache_dir', './cache'),
            n_mels=n_mels,
            lpf_cutoff_hz=cfg.data.get('lpf_cutoff_hz', 1500),
            n_freq_bins=cfg.data.get('n_freq_bins', None),
            use_mel_radar=use_mel_radar,
            n_mel_radar=n_mel_radar,
            mel_fmax=mel_fmax,
            mel_n_fft=mel_n_fft,
            dual_mel_radar=dual_mel_radar,
            use_spec_augment=cfg.data.get('spec_augment', True),
            normalize_radar_spec=cfg.data.get('normalize_radar_spec', False),
            teacher_model=cfg.model.teacher_model,
            use_masked_distill=cfg.trainer.get('use_masked_distill', False),
            mask_ratio=cfg.trainer.get('mask_ratio', 0.5),
            distillation_mode=cfg.trainer.get('distillation_mode', 'single'),
            teacher_layer_indices=cfg.model.get('teacher_layer_indices', None),
            source_teacher_layer_indices=cfg.model.get('source_teacher_layer_indices', None),
            cache_subdir_mode=cfg.trainer.get('cache_distillation_mode', None),
            vad_sidecar_dir=cfg.data.get('vad_sidecar_dir_train', None),
        )

        val_dataset = RADARdatasetCached(
            radar_paths=val_df['radar_path'].tolist(),
            audio_paths=val_df['audio_path'].tolist(),
            texts=val_df['text'].tolist(),
            tokenizer=tokenizer,
            device='cpu',
            train=False,
            cache_dir=cfg.data.get('cache_dir', './cache'),
            n_mels=n_mels,
            lpf_cutoff_hz=cfg.data.get('lpf_cutoff_hz', 1500),
            n_freq_bins=cfg.data.get('n_freq_bins', None),
            use_mel_radar=use_mel_radar,
            n_mel_radar=n_mel_radar,
            mel_fmax=mel_fmax,
            mel_n_fft=mel_n_fft,
            dual_mel_radar=dual_mel_radar,
            use_spec_augment=False,
            normalize_radar_spec=cfg.data.get('normalize_radar_spec', False),
            teacher_model=cfg.model.teacher_model,
            use_masked_distill=False,
            mask_ratio=0.0,
            distillation_mode=cfg.trainer.get('distillation_mode', 'single'),
            teacher_layer_indices=cfg.model.get('teacher_layer_indices', None),
            source_teacher_layer_indices=cfg.model.get('source_teacher_layer_indices', None),
            cache_subdir_mode=cfg.trainer.get('cache_distillation_mode', None),
            vad_sidecar_dir=cfg.data.get('vad_sidecar_dir_val', None),
        )
    else:
        print(">> Using on-the-fly dataset (slower loading)")
        train_dataset = RADARdataset(
            radar_paths=train_df['radar_path'].tolist(),
            audio_paths=train_df['audio_path'].tolist(),
            texts=train_df['text'].tolist(),
            device='cpu',
            train=True,
            n_mels=n_mels,
            lpf_cutoff_hz=cfg.data.get('lpf_cutoff_hz', 1500),  # Low-pass filter for radar signal
            use_spec_augment=cfg.data.get('spec_augment', True)  # Enable/disable spec augment
        )

        val_dataset = RADARdataset(
            radar_paths=val_df['radar_path'].tolist(),
            audio_paths=val_df['audio_path'].tolist(),
            texts=val_df['text'].tolist(),
            device='cpu',
            train=False,
            n_mels=n_mels,
            lpf_cutoff_hz=cfg.data.get('lpf_cutoff_hz', 1500),  # Low-pass filter for radar signal
            use_spec_augment=False  # Never augment validation
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.data.num_workers > 0 else False
    )
    
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""
    
    # Print config
    print("="*80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # A100 Optimization: Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')  # Options: 'medium' (default), 'high' (faster, slight precision loss)
    print("Set matmul precision to 'high' for A100 Tensor Cores")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(cfg)

    # Auto-detect teacher_dim based on teacher_model
    teacher_dims = {
        'tiny': 384,
        'base': 512,
        'small': 768,
        'medium': 1024,
        'large': 1280,
        'large-v2': 1280,
        'large-v3': 1280
    }
    teacher_model = cfg.model.get('teacher_model', 'tiny')
    if teacher_model in teacher_dims:
        cfg.model.teacher_dim = teacher_dims[teacher_model]
        print(f"Auto-detected teacher_dim={cfg.model.teacher_dim} for {teacher_model}")

    # Auto-calculate input_freq based on mel/linear spectrogram config
    from src.lightning.datamodule import calculate_freq_bins
    use_mel_radar = cfg.data.get('use_mel_radar', False)
    n_mel_radar = cfg.data.get('n_mel_radar', 80)
    n_freq_bins = cfg.data.get('n_freq_bins', None)
    lpf_cutoff_hz = cfg.data.get('lpf_cutoff_hz', None)

    if use_mel_radar:
        cfg.model.input_freq = n_mel_radar
        print(f"Using mel radar spectrogram: input_freq={n_mel_radar} mel bins (fmax={cfg.data.get('mel_fmax', 2500)} Hz)")
    elif n_freq_bins is not None:
        cfg.model.input_freq = n_freq_bins
        print(f"Using n_freq_bins={n_freq_bins} (direct bin count)")
    else:
        cfg.model.input_freq = calculate_freq_bins(lpf_cutoff_hz)
        if lpf_cutoff_hz is not None:
            print(f"Auto-detected input_freq={cfg.model.input_freq} for lpf_cutoff_hz={lpf_cutoff_hz} Hz")
        else:
            print(f"Using full spectrum: input_freq={cfg.model.input_freq} (no frequency cropping)")

    # Initialize model
    model = RadarLightningModule(cfg)
    
    # Setup WandB Logger
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_dir=os.path.join(HydraConfig.get().run.dir, "wandb")
    )
    
    # Callbacks
    pretrain_mode = cfg.trainer.get('pretrain_mode', None)

    callbacks = [
        # Best-by-monitor checkpoints (top-3 by val/wer)
        ModelCheckpoint(
            dirpath=os.path.join(HydraConfig.get().run.dir, "checkpoints"),
            filename='{epoch:02d}-{val_loss:.4f}',
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.mode,
            save_top_k=cfg.trainer.save_top_k,
            save_last=True,
            verbose=True
        ),
        # Latest-epoch checkpoint (always save, no metric filter).
        # Insurance: lets ckpt_path=max recover from a mid-training kill or wall-time
        # interrupt even when val/wer has plateaued and stopped entering top-k.
        # With monitor=None, Lightning only permits save_top_k in {-1, 0, 1};
        # save_top_k=1 keeps a single rolling latest-epoch ckpt.
        ModelCheckpoint(
            dirpath=os.path.join(HydraConfig.get().run.dir, "checkpoints"),
            filename='latest-{epoch:02d}',
            save_top_k=1,
            monitor=None,
            every_n_epochs=1,
            save_last=False,
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=100),
    ]

    # Only add decoder-dependent callbacks in end-to-end mode (not during CTC pretraining)
    if pretrain_mode is None:
        callbacks.extend([
            LogPredictionsCallback(
                tokenizer=whisper.tokenizer.get_tokenizer(False, task="transcribe"),
                num_display=8,
                num_save=50,
                save_every_n_epochs=5,
                output_dir=os.path.join(HydraConfig.get().run.dir, "transcriptions"),
            ),
        ])
        print("✓ Decoder callbacks enabled (end-to-end mode)")
    else:
        print(f"⚠ Skipping decoder callbacks (pretrain mode: {pretrain_mode})")

    # Encoder self-attention map visualization (only when enc_attn distillation is active)
    if cfg.trainer.get('w_enc_attn', 0.0) > 0:
        callbacks.append(LogAttnMapsCallback(log_every_n_epochs=5, num_samples=2))
        print("✓ Encoder attention map logging enabled (w_enc_attn > 0)")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.trainer.get('devices', 1),
        strategy=DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if cfg.trainer.get('strategy', 'auto') == 'ddp' else cfg.trainer.get('strategy', 'auto'),
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=False,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    print("\n" + "="*80)
    ckpt_path = cfg.trainer.get('ckpt_path', None)
    if ckpt_path == "max":
        # Auto-find latest checkpoint from previous runs with same experiment prefix
        # Derives pattern from hydra.run.dir: e.g. DEEP_6l_encoder_12345 -> DEEP_6l_encoder_*
        run_dir = HydraConfig.get().run.dir
        parent = os.path.dirname(run_dir)
        dirname = os.path.basename(run_dir)
        # Wildcard both job ID and task index: S2_CONF-MSE_44196_0 → S2_CONF-MSE_*_*
        prefix = re.sub(r'_\d+_\d+$', '_*_*', dirname)
        if prefix == dirname:
            # Fallback for single-number suffix: DEEP_6l_encoder_12345 → DEEP_6l_encoder_*
            prefix = re.sub(r'_\d+$', '_*', dirname)
        pattern = os.path.join(parent, prefix, "checkpoints", "last*.ckpt")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        # For job-ID dirs (e.g. S1_WSMALL_V2_467375_0 → prefix has wildcard),
        # exclude the current run since it won't have checkpoints yet.
        # For fixed dirs (e.g. S1_WSMALL_V2_MAIN → no substitution, prefix==dirname),
        # include the current dir so it finds its own checkpoints on resume.
        is_fixed_dir = (prefix == dirname)
        if not is_fixed_dir:
            matches = [m for m in matches if run_dir not in m]
        if matches:
            ckpt_path = matches[-1]
            print(f"'max' resolved: {pattern} -> {ckpt_path}")
        else:
            print(f"WARNING: 'max' found no checkpoints matching: {pattern}")
            ckpt_path = None
    elif ckpt_path and '*' in ckpt_path:
        # Glob pattern: find the newest matching run directory
        matches = sorted(glob.glob(ckpt_path), key=os.path.getmtime)
        if matches:
            ckpt_path = matches[-1]
            print(f"Glob resolved to: {ckpt_path}")
        else:
            print(f"WARNING: No matches for glob pattern: {ckpt_path}")
            ckpt_path = None
    if ckpt_path and os.path.isdir(ckpt_path):
        # Auto-resolve: if ckpt_path is a directory, find the best checkpoint
        ckpt_dir = ckpt_path
        if os.path.isdir(os.path.join(ckpt_dir, "checkpoints")):
            ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
        # Pick newest last*.ckpt (covers last.ckpt and last-v1.ckpt etc.)
        last_ckpts = sorted(
            glob.glob(os.path.join(ckpt_dir, "last*.ckpt")),
            key=os.path.getmtime
        )
        if last_ckpts:
            ckpt_path = last_ckpts[-1]
        else:
            ckpts = sorted(
                [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')],
                key=os.path.getmtime
            )
            ckpt_path = ckpts[-1] if ckpts else None
        print(f"Auto-resolved checkpoint: {ckpt_path}")
    if ckpt_path:
        print(f"Resuming Training from: {ckpt_path}")
    else:
        print("Starting Training...")
    print("="*80 + "\n")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    
    # Test with best model
    print("\n" + "="*80)
    print("Training Complete! Testing best model...")
    print("="*80 + "\n")
    
    trainer.test(model, val_loader, ckpt_path='best')
    
    print(f"\nBest model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"All outputs in: {HydraConfig.get().run.dir}")


if __name__ == "__main__":
    main()