import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss.distillation import create_distillation_strategy
import whisper


def temporal_attention_loss(student_h, teacher_h, mask, eps=1e-6):
    """
    Attention Transfer loss - matches temporal attention maps between student and teacher.
    Based on "Paying More Attention to Attention" (Zagoruyko & Komodakis, ICLR 2017).

    Computes attention as sum of |activations| over feature dimension (p=1),
    then matches L2-normalized attention maps.

    Args:
        student_h: Student hidden states [B, T, D_student]
        teacher_h: Teacher hidden states [B, T, D_teacher]
        mask: Padding mask [B, T] (1 for valid, 0 for padding)
        eps: Small constant for numerical stability

    Returns:
        Scalar AT loss value
    """
    # Ensure mask dtype matches activations
    mask = mask.to(student_h.dtype)

    # Compute temporal attention maps: sum of |activations| over feature dim
    # Using p=1 (abs) for robustness with negative activations
    s_attn = student_h.abs().sum(dim=-1)  # [B, T]
    t_attn = teacher_h.abs().sum(dim=-1)  # [B, T]

    # Zero out padding BEFORE normalizing (critical!)
    s_attn = s_attn * mask
    t_attn = t_attn * mask

    # L2 normalize per sample (only valid frames contribute to norm)
    s_attn = s_attn / (s_attn.norm(p=2, dim=-1, keepdim=True) + eps)
    t_attn = t_attn / (t_attn.norm(p=2, dim=-1, keepdim=True) + eps)

    # Masked L2 distance (explicit mask for safety)
    diff2 = ((s_attn - t_attn) ** 2) * mask  # [B, T]

    # Sum over frames (not mean!) to match original AT paper (Zagoruyko 2017)
    valid = (mask.sum(dim=-1) > 0).to(student_h.dtype)  # [B]
    loss_per_sample = diff2.sum(dim=-1)  # [B], squared L2 distance, bounded [0, 4]

    return (loss_per_sample * valid).sum() / (valid.sum() + eps)


class FeatureLoss(nn.Module):
    """
    1. Feature Alignment (MSE or Huber)
    Ensures Student features match Teacher features.
    Crucial: Applies masking to ignore padding.

    Loss types:
    - 'mse': Mean Squared Error (default) - sensitive to outliers
    - 'huber': Huber loss - robust to outliers, quadratic for small errors, linear for large
    """
    def __init__(self, loss_type='mse', delta=1.0):
        """
        Args:
            loss_type: 'mse' or 'huber'
            delta: Huber loss threshold (only used if loss_type='huber')
                   Errors smaller than delta use quadratic loss, larger use linear.
                   Default=1.0 is a reasonable starting point.
        """
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'mse' or 'huber'")

        print(f"FeatureLoss: Using {loss_type.upper()} loss" +
              (f" (delta={delta})" if loss_type == 'huber' else ""))

    def forward(self, student_feats, teacher_feats, padding_mask, temporal_window=None):
        """
        student_feats: [B, T, D]
        teacher_feats: [B, T, D]
        padding_mask:  [B, T] (1 for valid, 0 for pad)
        temporal_window: int or None. If set, randomly sample a contiguous window of
                         this many tokens each forward pass so all temporal positions
                         receive equal gradient pressure over training.
        """
        T = student_feats.shape[1]

        # Random temporal window: sample a contiguous slice each forward pass
        if temporal_window is not None and temporal_window < T and self.training:
            t_start = torch.randint(0, T - temporal_window + 1, (1,)).item()
            t_end = t_start + temporal_window
            student_feats = student_feats[:, t_start:t_end, :]
            teacher_feats = teacher_feats[:, t_start:t_end, :]
            padding_mask  = padding_mask[:, t_start:t_end]

        # Calculate element-wise loss (MSE or Huber)
        loss = self.loss_fn(student_feats, teacher_feats)  # [B, T', D]

        # Average across D dimension
        loss = loss.mean(dim=-1)  # [B, T']

        # Apply Mask (Multiplication broadcasts correctly)
        masked_loss = loss * padding_mask

        # Normalize by number of valid frames (avoid divide by zero)
        return masked_loss.sum() / (padding_mask.sum() + 1e-6)

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss.
    Maximizes cosine similarity between correct laser-audio pairs.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_emb, teacher_emb, padding_mask):
        # 1. Masked Global Pooling (Mean over valid frames only)
        mask_expanded = padding_mask.unsqueeze(-1)  # [B, T, 1]
        
        student_sum = (student_emb * mask_expanded).sum(dim=1)  # [B, D]
        teacher_sum = (teacher_emb * mask_expanded).sum(dim=1)  # [B, D]
        
        valid_frames = padding_mask.sum(dim=1, keepdim=True) + 1e-6  # [B, 1]
        
        student_global = student_sum / valid_frames
        teacher_global = teacher_sum / valid_frames

        # 2. Normalize
        student_global = F.normalize(student_global, p=2, dim=1)
        teacher_global = F.normalize(teacher_global, p=2, dim=1)

        # 3. Similarity Matrix
        logits = torch.matmul(student_global, teacher_global.T) / self.temperature
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # 4. Symmetric Loss
        loss_r2a = self.cross_entropy(logits, labels)
        loss_a2r = self.cross_entropy(logits.T, labels)

        return (loss_r2a + loss_a2r) / 2

class KDLoss(nn.Module):
    """
    2. Logit Distillation (KL Divergence) - FIXED
    Ensures Student's probability distribution matches Teacher's.
    Now correctly handles Masking.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # CHANGED: reduction='none' so we can mask manually
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, student_logits, teacher_logits, label_mask):
        """
        student_logits: [B, SeqLen, Vocab]
        teacher_logits: [B, SeqLen, Vocab]
        label_mask: [B, SeqLen] (True for valid tokens, False for padding)
        """
        # Cast to float32 to prevent BF16 underflow with large vocab (51865 tokens)
        # log_softmax in BF16 can produce -inf -> kl_div(-inf, p) = NaN
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        # Teacher: Standard Softmax
        target_dist = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Student: Log Softmax
        input_log_dist = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Calculate KL Divergence per token [B, SeqLen, Vocab]
        loss = self.kl_div(input_log_dist, target_dist)
        
        # Sum over Vocab dimension to get loss per token [B, SeqLen]
        loss = loss.sum(dim=-1)
        
        # Apply Mask (Zero out loss for padding tokens)
        loss = loss * label_mask
        
        # Normalize by number of valid tokens
        return loss.sum() / (label_mask.sum() + 1e-6) * (self.temperature ** 2)

class TaskLoss(nn.Module):
    """
    3. Hard Label Loss (Cross Entropy)
    """
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, student_logits, labels):
        # Cast to float32 to prevent BF16 overflow in cross-entropy with large vocab
        return self.ce(student_logits.float().transpose(1, 2), labels)

class CTCLoss(nn.Module):
    """
    CTC Loss for Alignment Learning
    Forces encoder to learn frame-level character predictions.
    Improves cross-attention alignment between encoder frames and decoder tokens.
    """
    def __init__(self, blank_id=0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

    def forward(self, ctc_logits, labels, input_lengths, target_lengths):
        """
        Args:
            ctc_logits: [B, T, vocab_size] - Frame-level predictions from encoder
            labels: [B, S] - Target token IDs
            input_lengths: [B] - Length of each encoder sequence (non-padded)
            target_lengths: [B] - Length of each target sequence (non-padded)

        Returns:
            CTC loss value
        """
        # CTC expects log probabilities
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Transpose to [T, B, vocab_size] (CTC expects time-major format)
        log_probs = log_probs.transpose(0, 1)

        return self.ctc(log_probs, labels, input_lengths, target_lengths)

class CompositeLoss(nn.Module):
    """
    Composite loss with KD warmup support and CTC for alignment learning.

    Supports multiple distillation modes:
    - single: Only final encoder output (default)
    - multi_layer: Intermediate block outputs + final
    - component_wise: Stem + Attention + Blocks + Final
    """
    def __init__(self, w_feat=10.0, w_kd=1.0, w_task=1.0, w_contra=1.0, w_ctc=1.0,
                 kd_warmup_start=None, kd_warmup_steps=0,
                 ctc_warmup_start=None, ctc_warmup_steps=0,
                 task_warmup_start=None, task_warmup_steps=0,
                 feat_loss_type='mse', feat_huber_delta=1.0,
                 ctc_layer2_weight=0.3, ctc_layer4_weight=0.7,
                 distillation_mode='single',
                 block_weights=None, component_weights=None,
                 teacher_layer_indices=None, num_student_layers=None,
                 use_attention_transfer=False, use_layer_norm=False,
                 at_weight=1.0, mse_weight=1.0, cosine_weight=0.0,
                 use_rkd=False, rkd_weight=0.35, rkd_n_samples=64, rkd_from_block=0, rkd_to_block=None,
                 kd_temperature=2.0,
                 distill_temporal_window=None, feature_norm='token'):
        """
        Args:
            feat_loss_type: 'mse' or 'huber' for feature alignment loss
            feat_huber_delta: Huber loss delta parameter (only used if feat_loss_type='huber')
            ctc_layer2_weight: Weight for layer 2 CTC loss (default 0.3). Set to 0.0 to disable.
            ctc_layer4_weight: Weight for layer 4 CTC loss (default 0.7). Should sum to 1.0 with layer2.
            distillation_mode: 'single', 'multi_layer', or 'component_wise'
            block_weights: List of weights for multi-layer blocks (default: auto-generated)
            component_weights: Dict of weights for component-wise distillation
            teacher_layer_indices: List of teacher block indices for layer mapping (multi-layer only)
            num_student_layers: Number of student layers (for auto-generating block weights)
            use_attention_transfer: Enable Attention Transfer loss (multi-layer only)
            use_layer_norm: Apply LayerNorm before MSE (multi-layer only)
            at_weight: Weight for AT loss relative to block weight
            mse_weight: Weight for MSE loss relative to block weight
            cosine_weight: Weight for cosine similarity loss relative to block weight
        """
        super().__init__()
        self.w_feat = w_feat
        self.w_kd = w_kd
        self.w_task = w_task
        self.w_contra = w_contra
        self.w_ctc = w_ctc
        self.distillation_mode = distillation_mode
        self.distill_temporal_window = distill_temporal_window  # None = use all tokens

        # KD Warmup
        self.kd_warmup_start = kd_warmup_start if kd_warmup_start is not None else w_kd
        self.kd_warmup_steps = kd_warmup_steps
        self.kd_target = w_kd
        self.current_kd_weight = self.kd_warmup_start

        # CTC Warmup (CRITICAL for stability - CTC loss is ~600-700 initially!)
        self.ctc_warmup_start = ctc_warmup_start if ctc_warmup_start is not None else w_ctc
        self.ctc_warmup_steps = ctc_warmup_steps
        self.ctc_target = w_ctc
        self.current_ctc_weight = self.ctc_warmup_start

        # Task (CE) Warmup — ramp decoder CE from near-zero so MSE anchor establishes latent
        # neighborhood before the decoder's cross-attention gradients take over
        self.task_warmup_start = task_warmup_start if task_warmup_start is not None else w_task
        self.task_warmup_steps = task_warmup_steps
        self.task_target = w_task
        self.current_task_weight = self.task_warmup_start

        # Multi-layer CTC weights
        self.ctc_layer2_weight = ctc_layer2_weight
        self.ctc_layer4_weight = ctc_layer4_weight

        # Create distillation strategy based on mode
        if distillation_mode == 'single':
            self.feat_loss_fn = None  # Use strategy instead
            self.distillation_strategy = create_distillation_strategy(
                'single',
                loss_type=feat_loss_type,
                delta=feat_huber_delta,
                use_attention_transfer=use_attention_transfer,
                use_layer_norm=use_layer_norm,
                at_weight=at_weight,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight
            )
        elif distillation_mode == 'multi_layer':
            self.feat_loss_fn = None  # Use strategy instead
            self.distillation_strategy = create_distillation_strategy(
                'multi_layer',
                loss_type=feat_loss_type,
                delta=feat_huber_delta,
                block_weights=block_weights,
                final_weight=1.0,
                teacher_layer_indices=teacher_layer_indices,
                num_student_layers=num_student_layers,
                use_attention_transfer=use_attention_transfer,
                use_layer_norm=use_layer_norm,
                at_weight=at_weight,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
                use_rkd=use_rkd,
                rkd_weight=rkd_weight,
                rkd_n_samples=rkd_n_samples,
                rkd_from_block=rkd_from_block,
                rkd_to_block=rkd_to_block,
                feature_norm=feature_norm,
            )
        elif distillation_mode == 'component_wise':
            self.feat_loss_fn = None  # Use strategy instead
            self.distillation_strategy = create_distillation_strategy(
                'component_wise',
                loss_type=feat_loss_type,
                delta=feat_huber_delta,
                component_weights=component_weights
            )
        else:
            raise ValueError(f"Unknown distillation_mode: {distillation_mode}")

        self.kd_loss_fn = KDLoss(temperature=kd_temperature)
        self.task_loss_fn = TaskLoss()
        self.contra_loss_fn = ContrastiveLoss()
        self.ctc_loss_fn = CTCLoss()

        # Initialize English-only tokenizer for CTC (decode tokens → text → characters)
        # CRITICAL: multilingual=False to avoid <|en|> and other language tokens
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, task="transcribe")

        # Print CTC layer configuration
        if self.w_ctc > 0:
            print(f"CTC Multi-Layer Weights: Layer2={ctc_layer2_weight:.1f}, Layer4={ctc_layer4_weight:.1f}")

    def update_kd_weight(self, current_step):
        if current_step >= self.kd_warmup_steps:
            self.current_kd_weight = self.kd_target
        else:
            progress = current_step / max(1, self.kd_warmup_steps)
            self.current_kd_weight = self.kd_warmup_start + (self.kd_target - self.kd_warmup_start) * progress
        return self.current_kd_weight

    def update_ctc_weight(self, current_step):
        """Update CTC weight with warmup (prevents gradient explosion)"""
        if current_step >= self.ctc_warmup_steps:
            self.current_ctc_weight = self.ctc_target
        else:
            progress = current_step / max(1, self.ctc_warmup_steps)
            self.current_ctc_weight = self.ctc_warmup_start + (self.ctc_target - self.ctc_warmup_start) * progress
        return self.current_ctc_weight

    def update_task_weight(self, current_step):
        """Update task (CE) weight with warmup — lets MSE anchor establish latent neighborhood first."""
        if current_step >= self.task_warmup_steps:
            self.current_task_weight = self.task_target
        else:
            progress = current_step / max(1, self.task_warmup_steps)
            self.current_task_weight = self.task_warmup_start + (self.task_target - self.task_warmup_start) * progress
        return self.current_task_weight

    def forward(self,
                student_feats, teacher_feats, padding_mask,
                student_logits, teacher_logits,
                labels,
                ctc_logits=None, input_lengths=None, target_lengths=None,
                student_feats_intermediate=None, teacher_feats_intermediate=None):
        """
        Args:
            student_feats: Final student features [B, T, D]
            teacher_feats: Final teacher features [B, T, D] (single mode) or Dict (multi-layer)
            padding_mask: [B, T] mask for valid frames
            student_logits: Decoder logits from student
            teacher_logits: Decoder logits from teacher
            labels: Ground truth labels
            ctc_logits: Optional tuple of (ctc_logits_layer2, ctc_logits_layer4) or single tensor
            input_lengths: Optional [B] encoder sequence lengths
            target_lengths: Optional [B] target sequence lengths
            student_feats_intermediate: Dict with 'blocks_projected' for multi-layer distillation
            teacher_feats_intermediate: List of teacher block outputs for multi-layer distillation

        Note: Handles pretrain modes where decoder outputs (student_logits, teacher_logits) are None
        """
        # Get device from student_feats
        device = student_feats.device

        # 1. Feature Loss (Uses padding_mask from Encoder)
        # Only compute if teacher_feats is not None and weight > 0
        l_feat = torch.tensor(0.0, device=device)
        distillation_losses = {}

        if self.w_feat > 0 and teacher_feats is not None:
            if self.distillation_strategy is not None:
                # Multi-layer or component-wise distillation
                # Build student outputs dict
                student_outputs = {'final': student_feats}
                if student_feats_intermediate is not None:
                    if 'blocks_projected' in student_feats_intermediate:
                        student_outputs['blocks'] = student_feats_intermediate['blocks_projected']
                    # Component-wise: add stem (projected to teacher_dim)
                    if 'stem_projected' in student_feats_intermediate:
                        student_outputs['stem'] = student_feats_intermediate['stem_projected']

                # Build teacher outputs dict
                if isinstance(teacher_feats, dict):
                    teacher_outputs = teacher_feats  # Already a dict with 'final', 'blocks', 'stem', etc.
                else:
                    teacher_outputs = {'final': teacher_feats}

                # Add intermediate blocks from separate argument if provided
                if teacher_feats_intermediate is not None:
                    teacher_outputs['blocks'] = teacher_feats_intermediate

                # Compute multi-layer losses
                distillation_losses = self.distillation_strategy(student_outputs, teacher_outputs, padding_mask,
                                                                  temporal_window=self.distill_temporal_window)
                l_feat = distillation_losses['total']

        # 2. KD Loss (only if decoder outputs exist)
        l_kd = torch.tensor(0.0, device=student_feats.device)
        if self.current_kd_weight > 0 and student_logits is not None and teacher_logits is not None:
            kd_mask = (labels != -100).float()
            l_kd = self.kd_loss_fn(student_logits, teacher_logits, kd_mask)

        # 3. Task Loss (only if decoder outputs exist)
        l_task = torch.tensor(0.0, device=student_feats.device)
        if self.w_task > 0 and student_logits is not None:
            l_task = self.task_loss_fn(student_logits, labels)

        # 4. Contrastive Loss (only if teacher exists)
        l_contra = torch.tensor(0.0, device=student_feats.device)
        if self.w_contra > 0 and teacher_feats is not None:
            teacher_feats_for_contra = teacher_feats['final'] if isinstance(teacher_feats, dict) else teacher_feats
            l_contra = self.contra_loss_fn(student_feats, teacher_feats_for_contra, padding_mask)

        # 6. Multi-Layer Character-Level CTC Loss (Y-Network)
        # Computes CTC loss on layers 2 and 4 with equal weights (0.5, 0.5)
        # Converts Whisper tokens → text → character indices (a-z, space, apostrophe)
        l_ctc = torch.tensor(0.0, device=student_feats.device)
        l_ctc_layer2 = torch.tensor(0.0, device=student_feats.device)
        l_ctc_layer4 = torch.tensor(0.0, device=student_feats.device)

        if ctc_logits is not None and input_lengths is not None:
            # Check if ctc_logits is a tuple (multi-layer) or single tensor (backward compat)
            is_multi_layer = isinstance(ctc_logits, tuple)

            # Convert Whisper token labels to text strings (using English-only tokenizer)
            texts = []
            for i in range(labels.shape[0]):
                # Remove padding (-100) and special tokens (>= 50256, including <|endoftext|>)
                valid_labels = labels[i][labels[i] != -100]
                text_only_labels = valid_labels[valid_labels < 50256]

                # Decode to text using self.tokenizer (English-only, no <|en|> tokens)
                try:
                    text = self.tokenizer.decode(text_only_labels.cpu().tolist())
                    texts.append(text)
                except:
                    texts.append("")  # Fallback for decode errors

            # Convert text to character-level CTC targets
            ctc_labels, target_lengths = text_to_ctc_targets(texts)
            ctc_labels = ctc_labels.to(device=labels.device)
            target_lengths = target_lengths.to(device=labels.device)

            # Debug: Print once
            if not hasattr(self, '_ctc_debug_printed'):
                print(f"\n[Character-Level CTC Debug - First Batch]")
                print(f"  Sample text: '{texts[0]}'")
                print(f"  Char targets (first 20): {ctc_labels[:20].cpu().tolist()}")
                print(f"  Target lengths (chars): {target_lengths.cpu().tolist()}")
                print(f"  Input lengths (frames): {input_lengths.cpu().tolist()}")
                print(f"  CTC vocab size: 29 (a-z + space + apostrophe + blank)")
                print(f"  Multi-layer CTC: {is_multi_layer}")
                if is_multi_layer:
                    print(f"  CTC weights: Layer 2 = {self.ctc_layer2_weight:.1f}, Layer 4 = {self.ctc_layer4_weight:.1f}")
                self._ctc_debug_printed = True

            if is_multi_layer:
                # Multi-layer CTC: Compute loss for layers 2 and 4
                ctc_logits_layer2, ctc_logits_layer4 = ctc_logits

                # Layer 2 CTC loss (early supervision) - only if weight > 0
                if ctc_logits_layer2 is not None and self.ctc_layer2_weight > 0:
                    l_ctc_layer2 = self.ctc_loss_fn(ctc_logits_layer2, ctc_labels, input_lengths, target_lengths)

                # Layer 4 CTC loss (final supervision) - only if weight > 0
                if ctc_logits_layer4 is not None and self.ctc_layer4_weight > 0:
                    l_ctc_layer4 = self.ctc_loss_fn(ctc_logits_layer4, ctc_labels, input_lengths, target_lengths)

                # Combine with configurable weights (default: 0.3 * layer2 + 0.7 * layer4)
                # Set ctc_layer2_weight=0.0 to disable layer 2 supervision
                l_ctc = self.ctc_layer2_weight * l_ctc_layer2 + self.ctc_layer4_weight * l_ctc_layer4
            else:
                # Backward compatibility: Single CTC head
                l_ctc = self.ctc_loss_fn(ctc_logits, ctc_labels, input_lengths, target_lengths)

        # Use current weights (with warmup)
        total = (self.w_feat * l_feat) + \
                (self.current_kd_weight * l_kd) + \
                (self.current_task_weight * l_task) + \
                (self.w_contra * l_contra) + \
                (self.current_ctc_weight * l_ctc)

        # Build loss dict
        loss_dict = {
            "loss_feat": l_feat,
            "loss_kd": l_kd,
            "loss_task": l_task,
            "loss_contra": l_contra,
            "loss_ctc": l_ctc,
            "loss_ctc_layer2": l_ctc_layer2,
            "loss_ctc_layer4": l_ctc_layer4
        }

        # Add per-block distillation losses for logging (if multi-layer mode)
        if distillation_losses:
            for key, value in distillation_losses.items():
                if key != 'total':  # Skip total, already in l_feat
                    loss_dict[f"loss_distill_{key}"] = value

        return total, loss_dict