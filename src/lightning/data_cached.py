"""
DISK-BASED CACHING: Pre-compute teacher embeddings and save to disk
No RAM limit - loads only what's needed per batch
"""

import torch
import torchaudio
import whisper
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

# Use same constants
from src.lightning.datamodule import (
    SAMPLE_RATE, CHUNK_LENGTH, N_SAMPLES, MAX_TEXT_LEN,
    N_FFT, HOP_LENGTH, WIN_LENGTH,
    load_and_pad_audio, get_laser_linear_spectrogram, get_laser_mel_spectrogram,
    tokenize_ground_truth
)


class LASERdatasetCached(Dataset):
    """
    Disk-based caching: Precomputes teacher embeddings and saves each sample as a separate file.

    Pros:
        - 10-50x faster training (precomputed teacher embeddings)
        - No RAM limit (loads samples on-demand)
        - Better GPU utilization

    Cons:
        - Slow first-time preprocessing (5-10 minutes for 26K samples)
        - Requires disk space (~20-40GB compressed)

    Usage:
        - Use for training with large datasets
        - Safe for any dataset size (no OOM issues)

    Distillation Modes:
        - single: Only final encoder output (default, ~65GB for large-v3)
        - multi_layer: Intermediate block outputs + final (~330GB for large-v3)
        - component_wise: Stem + attention + blocks + final (~400GB for large-v3)
    """

    def __init__(self, laser_paths, audio_paths, texts, tokenizer, device="cpu", train=False, cache_dir=None, n_mels=128, lpf_cutoff_hz=1500, n_freq_bins=None, use_mel_radar=False, n_mel_radar=80, mel_fmax=2500, mel_n_fft=1024, dual_mel_radar=False, use_spec_augment=True, normalize_radar_spec=False, teacher_model="large-v3", use_masked_distill=False, mask_ratio=0.5, distillation_mode="single", teacher_layer_indices=None, source_teacher_layer_indices=None, cache_subdir_mode=None, vad_sidecar_dir=None):
        self.laser_paths = laser_paths
        self.audio_paths = audio_paths
        self.texts = texts
        self.tokenizer = tokenizer  # Tokenizer passed from main process (avoid worker deadlock)
        self.device = device
        self.train = train
        self.n_mels = n_mels
        self.lpf_cutoff_hz = lpf_cutoff_hz  # Low-pass filter cutoff (default: 1500 Hz to filter noise)
        self.n_freq_bins = n_freq_bins  # Direct bin count (overrides lpf_cutoff_hz if set)
        self.use_mel_radar = use_mel_radar  # Use mel spectrogram for laser input
        self.n_mel_radar = n_mel_radar      # Number of mel bins to USE at runtime
        self.mel_fmax = mel_fmax            # Max frequency for laser mel filterbank
        self.mel_n_fft = mel_n_fft          # FFT window size (1024 for better freq resolution on low-SNR laser)
        self.dual_mel_radar = dual_mel_radar and use_mel_radar  # Cache both mel50 and mel80
        self.use_spec_augment = use_spec_augment
        self.normalize_radar_spec = normalize_radar_spec  # Per-sample normalization
        self.teacher_model = teacher_model
        self.use_masked_distill = use_masked_distill
        self.mask_ratio = mask_ratio  # NEW: Masking ratio (default 50%)
        self.distillation_mode = distillation_mode  # "single", "multi_layer", or "component_wise"
        self.teacher_layer_indices = teacher_layer_indices  # Which teacher blocks to use for training
        self.source_teacher_layer_indices = source_teacher_layer_indices  # Source cache layer indices (for reusing larger cache)
        # Sidecar VAD masks: parallel directory of {sample_idx:06d}.npy uint8[1500] files
        # produced by scripts/precompute_vad.py. None or missing files -> mask of ones (no masking).
        self.vad_sidecar_dir = Path(vad_sidecar_dir) if vad_sidecar_dir else None
        if self.vad_sidecar_dir is not None and not self.vad_sidecar_dir.exists():
            print(f"WARN: vad_sidecar_dir does not exist: {self.vad_sidecar_dir} -> VAD masks will fall back to all-ones")

        # Compute block mapping if reusing a larger cache
        # Example: 12-layer cache [2,5,7,10,13,15,18,21,23,26,29,31] -> 6-layer [5,10,15,21,26,31]
        # Mapping: [1, 3, 5, 7, 9, 11] (positions in cached list)
        self.cache_block_mapping = None
        if source_teacher_layer_indices is not None and teacher_layer_indices is not None:
            if source_teacher_layer_indices != teacher_layer_indices:
                # Build mapping from source cache positions to target blocks
                source_to_pos = {idx: pos for pos, idx in enumerate(source_teacher_layer_indices)}
                self.cache_block_mapping = [source_to_pos[idx] for idx in teacher_layer_indices]
                print(f"Using cache block mapping: {teacher_layer_indices} -> positions {self.cache_block_mapping}")

        # Choose augmentation strategy based on mode
        if self.train and self.use_masked_distill:
            # Masked Distillation: Aggressive masking for reconstruction task
            # Masks ~50% of time frames with large contiguous blocks
            print(f"Using Masked Distillation mode (mask_ratio={mask_ratio:.1%})")
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=150, p=mask_ratio, iid_masks=False),  # Large blocks
                torchaudio.transforms.FrequencyMasking(freq_mask_param=40, iid_masks=False),  # Some freq masking
            )
        elif self.train and self.use_spec_augment:
            # Regular SpecAugment: Light augmentation for robustness
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=40, p=0.2),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
            )
        else:
            self.spec_augment = None

        # Cache directory setup
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else Path("./cache").resolve()
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectory for this specific cache configuration
        if use_mel_radar and dual_mel_radar:
            freq_suffix = f"_dualmel_fmax{mel_fmax}nfft{mel_n_fft}"
        elif use_mel_radar:
            freq_suffix = f"_mel{n_mel_radar}fmax{mel_fmax}nfft{mel_n_fft}"
        elif n_freq_bins is not None:
            freq_suffix = f"_bins{n_freq_bins}"
        elif lpf_cutoff_hz is not None:
            freq_suffix = f"_lpf{lpf_cutoff_hz}"
        else:
            freq_suffix = "_nolpf"
        norm_suffix = "_norm" if normalize_radar_spec else "_nonorm"
        teacher_suffix = f"_{teacher_model}"
        split_name = 'train' if train else 'val'
        # Add distillation mode suffix (each mode has its own cache)
        # cache_subdir_mode overrides which cache folder to open (e.g. use multilayer6L cache
        # in single-layer training mode, to get correct padding_masks without loading blocks).
        subdir_mode = cache_subdir_mode if cache_subdir_mode is not None else distillation_mode
        if subdir_mode == "multi_layer":
            # Include layer indices in cache name for unique caching per config
            # If reusing a larger cache, use source layer count for naming
            if source_teacher_layer_indices:
                n_layers = len(source_teacher_layer_indices)
                distill_suffix = f"_multilayer{n_layers}L"
            elif teacher_layer_indices:
                n_layers = len(teacher_layer_indices)
                distill_suffix = f"_multilayer{n_layers}L"
            else:
                distill_suffix = "_multilayer"
        elif subdir_mode == "component_wise":
            distill_suffix = "_component"
        else:
            distill_suffix = ""
        cache_subdir_name = f"cache_{split_name}_{n_mels}mels{freq_suffix}{norm_suffix}{teacher_suffix}{distill_suffix}"
        self.cache_subdir = self.cache_dir / cache_subdir_name
        self.cache_subdir.mkdir(exist_ok=True)

        # Check if cache exists (look for completion marker)
        self.completion_marker = self.cache_subdir / "_COMPLETE"

        expected_samples = len(self.laser_paths)
        cache_ready = False

        if self.completion_marker.exists():
            cached_count = sum(1 for _ in self.cache_subdir.glob("sample_*.pt"))
            last_expected_file = self.cache_subdir / f"sample_{max(expected_samples - 1, 0):06d}.pt"

            if cached_count >= expected_samples and (expected_samples == 0 or last_expected_file.exists()):
                cache_ready = True
                print(f"✓ Found cached dataset in {self.cache_subdir}")
                print(f"  Total samples: {expected_samples}")
            else:
                print(f"⚠ Incomplete cache detected in {self.cache_subdir}")
                print(f"  Expected {expected_samples} samples, found {cached_count}")
                print("  Rebuilding missing cache files...")

        if not cache_ready:
            print(f"Preprocessing dataset with precomputed teacher embeddings...")
            print(f"This will take 5-10 minutes for {expected_samples} samples")
            self._preprocess_all()
            # Mark cache as complete
            self.completion_marker.touch()
            print(f"✓ Cache saved to {self.cache_subdir}")

    def _preprocess_all(self):
        """Precompute teacher embeddings and save each sample individually (no RAM buildup)"""
        # Load teacher encoder ONCE for preprocessing
        preprocess_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the Whisper teacher
        print(f"Loading Whisper {self.teacher_model} encoder for preprocessing...")
        teacher = whisper.load_model(self.teacher_model)
        teacher.eval()
        teacher = teacher.to(preprocess_device)
        print(f"Whisper encoder loaded on {preprocess_device}")

        with torch.no_grad():  # No gradients needed for preprocessing
            for i in tqdm(range(len(self.laser_paths)), desc="Preprocessing"):
                cache_file = self.cache_subdir / f"sample_{i:06d}.pt"

                # Skip if already cached
                if cache_file.exists():
                    continue

                # Load audio
                laser_wav, r_len = load_and_pad_audio(self.laser_paths[i])
                audio_wav, a_len = load_and_pad_audio(self.audio_paths[i])

                # Compute laser spectrogram (always needed for student input)
                if self.dual_mel_radar:
                    # Store both mel50 and mel80 in one file — shared cache for both students
                    laser_spec_mel50 = get_laser_mel_spectrogram(
                        laser_wav, n_mels=50, fmax=self.mel_fmax,
                        n_fft=self.mel_n_fft, normalize=self.normalize_radar_spec
                    )
                    laser_spec_mel80 = get_laser_mel_spectrogram(
                        laser_wav, n_mels=80, fmax=self.mel_fmax,
                        n_fft=self.mel_n_fft, normalize=self.normalize_radar_spec
                    )
                    laser_spec = None  # not used in dual mode
                elif self.use_mel_radar:
                    laser_spec = get_laser_mel_spectrogram(
                        laser_wav,
                        n_mels=self.n_mel_radar,
                        fmax=self.mel_fmax,
                        n_fft=self.mel_n_fft,
                        normalize=self.normalize_radar_spec
                    )
                else:
                    laser_spec = get_laser_linear_spectrogram(
                        laser_wav,
                        lpf_cutoff_hz=self.lpf_cutoff_hz,
                        n_freq_bins=self.n_freq_bins,
                        normalize=self.normalize_radar_spec
                    )

                # PRECOMPUTE teacher encoder embeddings based on teacher type
                # For multi_layer: extract intermediate block outputs
                # For component_wise: extract stem + attention + blocks
                extract_intermediate = self.distillation_mode == "multi_layer"
                extract_components = self.distillation_mode == "component_wise"

                # Whisper processing
                teacher_mel = whisper.log_mel_spectrogram(audio_wav[0], n_mels=self.n_mels)  # [n_mels, 3000]
                teacher_mel_batch = teacher_mel.unsqueeze(0).to(preprocess_device)  # [1, n_mels, 3000]

                # Initialize outputs
                block_outputs = None
                stem_output = None
                attention_outputs = None

                if extract_components:
                    # Component-wise: extract stem + attention + blocks
                    teacher_feats, block_outputs, stem_output, attention_outputs = \
                        self._extract_whisper_components(teacher, teacher_mel_batch)
                elif extract_intermediate:
                    # Multi-layer: extract blocks only
                    teacher_feats, block_outputs = self._extract_whisper_intermediates(
                        teacher, teacher_mel_batch
                    )
                else:
                    teacher_feats = teacher.encoder(teacher_mel_batch)  # [1, 1500, teacher_dim]

                # Padding mask (1500 frames to match encoder output after stride=2)
                valid_frames = int((r_len / N_SAMPLES) * 1500)
                padding_mask = torch.zeros(1500)
                padding_mask[:valid_frames] = 1.0

                # Save this sample to disk immediately (no RAM buildup!)
                if self.dual_mel_radar:
                    cache_data = {
                        'laser_spec_mel50': laser_spec_mel50.cpu().half(),
                        'laser_spec_mel80': laser_spec_mel80.cpu().half(),
                        'teacher_feats': teacher_feats.squeeze(0).cpu(),
                        'padding_mask': padding_mask.cpu().bool()
                    }
                else:
                    cache_data = {
                        'laser_spec': laser_spec.cpu().half(),
                        'teacher_feats': teacher_feats.squeeze(0).cpu(),
                        'padding_mask': padding_mask.cpu().bool()
                    }

                # Add intermediate block outputs for multi-layer distillation
                if block_outputs is not None:
                    # Prefer source indices when provided so multiple students can reuse one cache.
                    if self.source_teacher_layer_indices is not None:
                        indices = self.source_teacher_layer_indices
                        selected_blocks = [block_outputs[i].squeeze(0).cpu().half() for i in indices]
                    # Otherwise cache only the requested teacher blocks.
                    elif self.teacher_layer_indices is not None:
                        indices = self.teacher_layer_indices
                        selected_blocks = [block_outputs[i].squeeze(0).cpu().half() for i in indices]
                    else:
                        # Cache all blocks (backward compatible)
                        indices = list(range(len(block_outputs)))
                        selected_blocks = [b.squeeze(0).cpu().half() for b in block_outputs]
                    cache_data['teacher_blocks'] = selected_blocks

                    # For component-wise: also save attention outputs (same block selection)
                    if attention_outputs is not None:
                        selected_attention = [attention_outputs[i].squeeze(0).cpu() for i in indices]
                        cache_data['teacher_attention'] = selected_attention

                # Add stem output for component-wise distillation
                if stem_output is not None:
                    cache_data['teacher_stem'] = stem_output.squeeze(0).cpu()

                torch.save(cache_data, cache_file)

        # Free teacher encoder memory
        del teacher
        torch.cuda.empty_cache()
        print("Teacher encoder released from memory")

    def _extract_whisper_intermediates(self, model, mel):
        """
        Extract intermediate block outputs from Whisper encoder.

        Whisper encoder structure:
        - conv1: Conv1d (n_mels -> n_state)
        - conv2: Conv1d (n_state -> n_state, stride=2) -> reduces 3000->1500
        - positional_embedding: [1500, n_state]
        - blocks: ModuleList of ResidualAttentionBlock

        Args:
            model: Whisper model
            mel: [1, n_mels, 3000]

        Returns:
            final_output: [1, 1500, n_state]
            block_outputs: List of [1, 1500, n_state] for each block
        """
        encoder = model.encoder

        # Run through conv stem
        x = torch.nn.functional.gelu(encoder.conv1(mel))
        x = torch.nn.functional.gelu(encoder.conv2(x))  # [1, n_state, 1500]
        x = x.permute(0, 2, 1)  # [1, 1500, n_state]

        # Add positional embedding
        x = (x + encoder.positional_embedding).to(x.dtype)

        # Run through transformer blocks and collect outputs
        block_outputs = []
        for block in encoder.blocks:
            x = block(x)
            block_outputs.append(x.clone())

        # Final layer norm
        x = encoder.ln_post(x)

        return x, block_outputs

    def _extract_whisper_components(self, model, mel):
        """
        Extract component-wise outputs from Whisper encoder.

        Components extracted:
        - stem: Output after conv layers + positional embedding (before blocks)
        - attention: Output after attention (before MLP) in each block
        - blocks: Full block outputs (after attention + MLP)
        - final: Final encoder output after layer norm

        Args:
            model: Whisper model
            mel: [1, n_mels, 3000]

        Returns:
            final_output: [1, 1500, n_state]
            block_outputs: List of [1, 1500, n_state] for each block
            stem_output: [1, 1500, n_state] after conv + positional embedding
            attention_outputs: List of [1, 1500, n_state] attention outputs per block
        """
        encoder = model.encoder

        # Run through conv stem
        x = torch.nn.functional.gelu(encoder.conv1(mel))
        x = torch.nn.functional.gelu(encoder.conv2(x))  # [1, n_state, 1500]
        x = x.permute(0, 2, 1)  # [1, 1500, n_state]

        # Add positional embedding
        x = (x + encoder.positional_embedding).to(x.dtype)

        # Save stem output (after conv + pos emb, before blocks)
        stem_output = x.clone()

        # Run through transformer blocks and collect outputs
        block_outputs = []
        attention_outputs = []

        for block in encoder.blocks:
            # Whisper ResidualAttentionBlock forward:
            # x = x + attn(attn_ln(x))
            # x = x + mlp(mlp_ln(x))

            # Extract attention output (before MLP)
            # Note: block.attn() returns (output, attention_weights) tuple
            attn_result = block.attn(block.attn_ln(x))
            if isinstance(attn_result, tuple):
                attn_result = attn_result[0]  # Extract just the output tensor
            attn_out = x + attn_result
            attention_outputs.append(attn_out.clone())

            # Complete block forward
            x = attn_out + block.mlp(block.mlp_ln(attn_out))
            block_outputs.append(x.clone())

        # Final layer norm
        x = encoder.ln_post(x)

        return x, block_outputs, stem_output, attention_outputs

    def __len__(self):
        return len(self.laser_paths)

    def __getitem__(self, idx):
        # Load precomputed data from disk (only this sample!)
        cache_file = self.cache_subdir / f"sample_{idx:06d}.pt"
        cached_data = torch.load(cache_file, map_location='cpu')

        # Extract tensors and convert to float32
        # Dual-mel cache: pick the spec matching this model's n_mel_radar
        dual_key = f'laser_spec_mel{self.n_mel_radar}'
        if dual_key in cached_data:
            laser_spec = cached_data[dual_key].clone().float()
        else:
            laser_spec = cached_data['laser_spec'].clone().float()  # backward compat
        teacher_feats = cached_data['teacher_feats'].clone()  # Already float32 - no conversion
        padding_mask = cached_data['padding_mask'].clone().float()  # Convert bool -> float32

        # Sanitize NaN/Inf in cached data (can occur from corrupted cache entries)
        laser_spec = torch.nan_to_num(laser_spec, nan=0.0, posinf=0.0, neginf=0.0)
        teacher_feats = torch.nan_to_num(teacher_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # The old cache stored laser_spec as [1, F, T] (3D). Some newer caches
        # (e.g. recache_zeroed_vad.py before the fix) stored it as [F, T] (2D).
        # Student encoder expects [B, 1, F, T] after batching, so per-sample we
        # need [1, F, T]. Add the channel dim if it's missing.
        if laser_spec.ndim == 2:
            laser_spec = laser_spec.unsqueeze(0)

        # Apply augmentation (only during training)
        if self.spec_augment is not None:
            laser_spec = self.spec_augment(laser_spec)

        # Tokenize text
        dec_input_ids, label_ids = tokenize_ground_truth(self.texts[idx], self.tokenizer)

        # Load sidecar VAD mask if available; fall back to ones (no masking).
        # uint8[1500] -> float[1500], same length as padding_mask after stride-2.
        if self.vad_sidecar_dir is not None:
            vad_file = self.vad_sidecar_dir / f"sample_{idx:06d}.npy"
            if vad_file.exists():
                vad_mask = torch.from_numpy(np.load(vad_file)).float()
            else:
                vad_mask = torch.ones(1500, dtype=torch.float32)
        else:
            vad_mask = torch.ones(1500, dtype=torch.float32)

        # Build return dict
        result = {
            "laser_spec": laser_spec,
            "decoder_input": torch.tensor(dec_input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "padding_mask": padding_mask,
            "vad_mask": vad_mask,
        }

        # For multi-layer distillation: return teacher_feats as dict with blocks
        if 'teacher_blocks' in cached_data and self.distillation_mode in ["multi_layer", "component_wise"]:
            # Load only needed blocks (saves memory when reusing larger cache)
            if self.cache_block_mapping is not None:
                # Only load the 6 blocks we need from 12-block cache
                blocks = [cached_data['teacher_blocks'][pos].clone().float() for pos in self.cache_block_mapping]
            else:
                # Load all blocks
                blocks = [b.clone().float() for b in cached_data['teacher_blocks']]

            teacher_dict = {
                'final': teacher_feats,
                'blocks': blocks
            }

            # For component-wise: also include stem and attention
            if self.distillation_mode == "component_wise":
                if 'teacher_stem' in cached_data:
                    teacher_dict['stem'] = cached_data['teacher_stem'].clone()
                if 'teacher_attention' in cached_data:
                    teacher_dict['attention'] = [a.clone() for a in cached_data['teacher_attention']]

            result["teacher_feats"] = teacher_dict
        else:
            # Single-layer mode: return tensor directly (backward compatible)
            result["teacher_feats"] = teacher_feats

        # Return CPU tensors - PyTorch Lightning will move them to GPU automatically
        return result
