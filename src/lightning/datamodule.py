import torch
import torchaudio
import whisper
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# --- Constants ---
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480,000 samples
MAX_TEXT_LEN = 448 # Standard Whisper Context Length

# STFT Parameters
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400

# --- Helper Functions ---

def calculate_freq_bins(lpf_cutoff_hz=None):
    """
    Calculate the number of frequency bins based on low-pass filter cutoff.

    Args:
        lpf_cutoff_hz: Low-pass filter cutoff frequency in Hz. If None, returns full spectrum bins.

    Returns:
        Number of frequency bins after cropping (or full bins if no cutoff)
    """
    full_bins = (N_FFT // 2) + 1  # 201 bins for N_FFT=400

    if lpf_cutoff_hz is None:
        return full_bins

    # Calculate cutoff bin based on frequency
    cutoff_bin = int((lpf_cutoff_hz / (SAMPLE_RATE / 2)) * full_bins)
    return cutoff_bin

def load_and_pad_audio(file_path):
    """
    Loads audio, resamples to 16kHz, and pads/trims to exactly 30 seconds.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = transform(waveform)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        num_frames = waveform.shape[1]
        if num_frames > N_SAMPLES:
            waveform = waveform[:, :N_SAMPLES]
            valid_length = N_SAMPLES
        else:
            padding = N_SAMPLES - num_frames
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            valid_length = num_frames
            
        return waveform, valid_length

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return torch.zeros(1, N_SAMPLES), 0

def get_radar_linear_spectrogram(waveform, lpf_cutoff_hz=None, n_freq_bins=None, normalize=False):
    """
    Calculates Linear STFT for the Student (Radar).

    Args:
        waveform: Input waveform
        lpf_cutoff_hz: Low-pass filter cutoff in Hz (alternative to n_freq_bins)
        n_freq_bins: Direct number of frequency bins to keep (overrides lpf_cutoff_hz if set)
        normalize: Whether to apply per-sample normalization
    """
    window = torch.hann_window(WIN_LENGTH).to(waveform.device)

    # Compute STFT
    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(stft)

    # Frequency Cropping (priority: n_freq_bins > lpf_cutoff_hz)
    if n_freq_bins is not None:
        # Direct bin count (most explicit control)
        if n_freq_bins < magnitude.shape[1]:
            magnitude = magnitude[:, :n_freq_bins, :]
    elif lpf_cutoff_hz is not None:
        # Frequency-based filtering
        freq_resolution = SAMPLE_RATE / N_FFT
        cutoff_bin = int(lpf_cutoff_hz / freq_resolution)

        if cutoff_bin < magnitude.shape[1]:
            magnitude = magnitude[:, :cutoff_bin, :]

    # Trim to 3000 frames
    if magnitude.shape[-1] > 3000:
        magnitude = magnitude[..., :3000]

    # Log Normalization
    log_spec = torch.log1p(magnitude)

    # Per-sample Normalization
    if normalize:
        mean = log_spec.mean()
        std = log_spec.std()
        log_spec = (log_spec - mean) / (std + 1e-6)

    return log_spec


def get_radar_mel_spectrogram(waveform, n_mels=80, fmax=2500, n_fft=1024, normalize=False):
    """
    Mel spectrogram for radar input. Uses larger n_fft than Whisper default (400)
    for better frequency resolution on low-SNR radar signals. hop_length stays
    at 160 to keep 3000 frames → 1500 after Whisper conv downsampling.

    Args:
        waveform: [1, N_SAMPLES] input waveform
        n_mels: number of mel bins (default 80)
        fmax: max frequency for mel filterbank in Hz (default 2500)
        n_fft: FFT window size (default 1024, ~64ms for better freq resolution)
        normalize: per-sample normalization

    Returns:
        [1, n_mels, 3000] log-mel spectrogram
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=n_fft,
        n_mels=n_mels,
        f_min=0.0,
        f_max=fmax,
        power=2.0,
    ).to(waveform.device)

    mel = mel_transform(waveform)  # [1, n_mels, T]

    # Trim to 3000 frames
    if mel.shape[-1] > 3000:
        mel = mel[..., :3000]

    # Whisper-style normalization: log10, clip to 8dB dynamic range, scale to [-1, ~1]
    log_mel = torch.clamp(mel, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    if normalize:
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-6)

    return log_mel


def tokenize_ground_truth(text, tokenizer):
    """
    Converts text to Whisper BPE tokens.
    """
    import re
    text = re.sub(r'<\|[^|]*\|>', '', text).strip()  # strip any Whisper special tokens from GT text
    text_tokens = tokenizer.encode(text, disallowed_special=())
    
    # For small.en (English-only), correct initial sequence is sot + no_timestamps.
    # The multilingual-style <|transcribe|> token is wrong here and degrades decoder
    # behavior on hard content. Old ckpts trained with the buggy 3-token prefix will
    # see a small distribution shift when evaluated with this corrected prefix.
    sot_sequence = list(tokenizer.sot_sequence_including_notimestamps)
    
    decoder_input = sot_sequence + text_tokens
    labels = sot_sequence[1:] + text_tokens + [tokenizer.eot]
    
    if len(decoder_input) < MAX_TEXT_LEN:
        decoder_input += [tokenizer.eot] * (MAX_TEXT_LEN - len(decoder_input))
    else:
        decoder_input = decoder_input[:MAX_TEXT_LEN]
        
    if len(labels) < MAX_TEXT_LEN:
        labels += [-100] * (MAX_TEXT_LEN - len(labels))
    else:
        labels = labels[:MAX_TEXT_LEN]
        
    return decoder_input, labels

# --- Main Dataset Class ---

class RADARdataset(Dataset):
    def __init__(self, radar_paths, teacher_feat_paths, texts, device="cpu", train=False,
                 lpf_cutoff_hz=None, n_freq_bins=None, use_spec_augment=True, normalize_radar_spec=False):
        """
        Args:
            radar_paths: List of file paths for radar wavs
            teacher_feat_paths: List of paths to .pt files (Pre-computed Teacher Features)
            texts: List of ground truth strings
            lpf_cutoff_hz: Low-pass filter cutoff in Hz (alternative to n_freq_bins)
            n_freq_bins: Direct number of frequency bins (overrides lpf_cutoff_hz if set)
            normalize_radar_spec: Whether to apply per-sample normalization (zero mean, unit variance)
        """
        self.radar_paths = radar_paths
        self.teacher_feat_paths = teacher_feat_paths
        self.texts = texts
        self.device = device
        self.train = train
        self.lpf_cutoff_hz = lpf_cutoff_hz
        self.n_freq_bins = n_freq_bins
        self.use_spec_augment = use_spec_augment
        self.normalize_radar_spec = normalize_radar_spec

        # Initialize Tokenizer (English-only)
        self.tokenizer = whisper.tokenizer.get_tokenizer(False, task="transcribe")
        # Augmentation
        if self.train and self.use_spec_augment:
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=40, p=0.2),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
            )
        else:
            self.spec_augment = None

    def __len__(self):
        return len(self.radar_paths)

    def __getitem__(self, idx):
        # 1. Load Radar (On the fly)
        radar_wav, r_len = load_and_pad_audio(self.radar_paths[idx])
        
        # 2. Compute Radar Spectrogram
        radar_spec = get_radar_linear_spectrogram(
            radar_wav,
            lpf_cutoff_hz=self.lpf_cutoff_hz,
            n_freq_bins=self.n_freq_bins,
            normalize=self.normalize_radar_spec
        )
        
        # 3. Load PRE-COMPUTED Teacher Features
        # CHANGED: We load the .pt file directly. 
        # The .pt file is a dict: {'radar_spec': ..., 'teacher_feats': ..., 'padding_mask': ...}
        # or a direct tensor from older caching code
        cached_data = torch.load(self.teacher_feat_paths[idx], map_location='cpu')
        
        # Extract teacher features (handle both dict and direct tensor formats)
        if isinstance(cached_data, dict):
            teacher_feats = cached_data['teacher_feats'].clone()
        else:
            teacher_feats = cached_data

        # 4. Apply Augmentation (Radar Only)
        if self.spec_augment is not None:
            radar_spec = self.spec_augment(radar_spec)

        # 5. Tokenize Text
        dec_input_ids, label_ids = tokenize_ground_truth(self.texts[idx], self.tokenizer)
        
        # 6. Create Padding Mask
        valid_frames = int((r_len / N_SAMPLES) * 1500)
        padding_mask = torch.zeros(1500)
        padding_mask[:valid_frames] = 1.0
        
        return {
            "radar_spec": radar_spec.to(self.device),       
            "teacher_feats": teacher_feats.to(self.device), 
            "decoder_input": torch.tensor(dec_input_ids, dtype=torch.long).to(self.device), 
            "labels": torch.tensor(label_ids, dtype=torch.long).to(self.device),            
            "padding_mask": padding_mask.to(self.device)    
        }

class RadarDataModule(pl.LightningDataModule):
    def __init__(self, train_radar_paths, train_feat_paths, train_texts, 
                 val_radar_paths, val_feat_paths, val_texts, 
                 batch_size=8, num_workers=4, **kwargs):
        super().__init__()
        # Expects feature paths now, not audio paths
        self.train_data = (train_radar_paths, train_feat_paths, train_texts)
        self.val_data = (val_radar_paths, val_feat_paths, val_texts)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RADARdataset(*self.train_data, train=True, **self.kwargs)
            self.val_dataset = RADARdataset(*self.val_data, train=False, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=4 if self.num_workers > 0 else None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0,
                          prefetch_factor=4 if self.num_workers > 0 else None)