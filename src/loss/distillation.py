"""
Modular Knowledge Distillation Strategies

Supports multiple distillation approaches:
- Single-layer: Final encoder output only (current default)
- Multi-layer: Multiple block outputs
- Component-wise: Stem + Attention + Blocks + Final
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class DistillationStrategy(nn.Module):
    """Base class for distillation strategies"""

    def __init__(self, loss_type='mse', delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Per-output-dim MSE weights. When set, the [B, T, D] elementwise loss is
        # collapsed over D by a weighted sum instead of a uniform mean. Used to
        # down-weight teacher-output dims that are driven by mel bins the student
        # cannot reconstruct (see scripts/calibrate_dim_weights.py). None = uniform.
        self.dim_weights = None

    def set_dim_weights(self, dim_weights):
        """dim_weights: 1D tensor of length D, or None to disable."""
        if dim_weights is None:
            self.dim_weights = None
            return
        if not isinstance(dim_weights, torch.Tensor):
            dim_weights = torch.as_tensor(dim_weights)
        self.dim_weights = dim_weights.float()

    def _collapse_over_dim(self, loss_3d):
        """[B, T, D] -> [B, T]. Uniform mean by default, weighted mean when dim_weights set."""
        if self.dim_weights is None:
            return loss_3d.mean(dim=-1)
        dw = self.dim_weights.to(loss_3d.device).view(1, 1, -1)  # [1, 1, D]
        return (loss_3d * dw).sum(dim=-1) / (dw.sum() + 1e-6)

    def compute_masked_loss(self, student_feat, teacher_feat, padding_mask):
        """Helper to compute masked loss between two features"""
        loss = self.loss_fn(student_feat, teacher_feat)  # [B, T, D]
        loss = self._collapse_over_dim(loss)  # [B, T] — dim-weighted when configured
        masked_loss = loss * padding_mask
        return masked_loss.sum() / (padding_mask.sum() + 1e-6)

    def forward(self, student_outputs, teacher_outputs, padding_mask):
        """Must be implemented by subclasses"""
        raise NotImplementedError


class SingleLayerDistillation(DistillationStrategy):
    """
    Single-layer: Only distill final encoder output.
    Supports AT, cosine similarity, and layer norm (same as MultiLayer but final only).

    Usage:
        student_outputs = {'final': [B, T, D]}
        teacher_outputs = {'final': [B, T, D]}
    """

    def __init__(self, loss_type='mse', delta=1.0,
                 use_attention_transfer=False, use_layer_norm=False,
                 at_weight=1.0, mse_weight=1.0, cosine_weight=0.0):
        super().__init__(loss_type, delta)
        self.use_attention_transfer = use_attention_transfer
        self.use_layer_norm = use_layer_norm
        self.at_weight = at_weight
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

        print(f"Distillation Strategy: Single-Layer (Final output only)")
        print(f"  MSE weight: {mse_weight} (layer_norm={use_layer_norm})")
        if use_attention_transfer:
            print(f"  Attention Transfer: weight={at_weight}")
        if cosine_weight > 0:
            print(f"  Cosine Similarity: weight={cosine_weight}")

    def compute_at_loss(self, student_h, teacher_h, mask, eps=1e-6):
        """Attention Transfer loss (same as MultiLayer)."""
        mask = mask.to(student_h.dtype)
        s_attn = student_h.abs().sum(dim=-1)
        t_attn = teacher_h.abs().sum(dim=-1)
        s_attn = s_attn * mask
        t_attn = t_attn * mask
        s_attn = s_attn / (s_attn.norm(p=2, dim=-1, keepdim=True) + eps)
        t_attn = t_attn / (t_attn.norm(p=2, dim=-1, keepdim=True) + eps)
        diff2 = ((s_attn - t_attn) ** 2) * mask
        valid = (mask.sum(dim=-1) > 0).to(student_h.dtype)
        loss_per_sample = diff2.sum(dim=-1)
        return (loss_per_sample * valid).sum() / (valid.sum() + eps)

    def compute_cosine_loss(self, student_feat, teacher_feat, padding_mask, eps=1e-6):
        """Frame-wise cosine similarity loss: 1 - cos_sim."""
        cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
        loss = 1.0 - cos_sim
        masked_loss = loss * padding_mask
        return masked_loss.sum() / (padding_mask.sum() + eps)

    def compute_normalized_mse_loss(self, student_feat, teacher_feat, padding_mask):
        """MSE with LayerNorm applied to both sides (computed in float32 to avoid BF16 overflow)."""
        s_norm = F.layer_norm(student_feat.float(), [student_feat.size(-1)]).to(student_feat.dtype)
        t_norm = F.layer_norm(teacher_feat.float(), [teacher_feat.size(-1)]).to(teacher_feat.dtype)
        loss = self.loss_fn(s_norm, t_norm)
        loss = self._collapse_over_dim(loss)  # dim-weighted when configured
        masked_loss = loss * padding_mask
        return masked_loss.sum() / (padding_mask.sum() + 1e-6)

    def forward(self, student_outputs, teacher_outputs, padding_mask, temporal_window=None):
        """
        Args:
            student_outputs: Dict with key 'final' -> [B, T, D]
            teacher_outputs: Dict with key 'final' -> [B, T, D]
            padding_mask: [B, T]
            temporal_window: int or None. If set, randomly slice a contiguous T-window
                             from final + padding_mask before computing losses.
                             Matches MultiLayerDistillation behaviour.
        """
        student_final = student_outputs['final']
        teacher_final = teacher_outputs['final']

        if temporal_window is not None:
            T = padding_mask.shape[1]
            if temporal_window < T:
                t_start = torch.randint(0, T - temporal_window + 1, (1,)).item()
                t_end = t_start + temporal_window
                student_final = student_final[:, t_start:t_end]
                teacher_final = teacher_final[:, t_start:t_end]
                padding_mask = padding_mask[:, t_start:t_end]
        losses = {}
        total_loss = 0

        # MSE (optionally with LayerNorm)
        if self.use_layer_norm:
            mse_loss = self.compute_normalized_mse_loss(student_final, teacher_final, padding_mask)
        else:
            mse_loss = self.compute_masked_loss(student_final, teacher_final, padding_mask)
        losses['final_mse'] = mse_loss
        losses['mse_total'] = mse_loss
        total_loss += mse_loss * self.mse_weight

        # AT
        if self.use_attention_transfer:
            at_loss = self.compute_at_loss(student_final, teacher_final, padding_mask)
            losses['final_at'] = at_loss
            losses['at_total'] = at_loss
            total_loss += at_loss * self.at_weight

        # Cosine
        if self.cosine_weight > 0:
            cosine_loss = self.compute_cosine_loss(student_final, teacher_final, padding_mask)
            losses['final_cosine'] = cosine_loss
            losses['cosine_total'] = cosine_loss
            total_loss += cosine_loss * self.cosine_weight

        losses['total'] = total_loss
        return losses


class MultiLayerDistillation(DistillationStrategy):
    """
    Multi-layer: Distill intermediate block outputs + final

    Supports layer mapping for different student/teacher depths:
    - teacher_layer_indices: Select specific teacher blocks to match student blocks
    - Example: 12 student blocks -> 12 selected teacher blocks from 32 total

    NEW: Supports Attention Transfer (AT) loss alongside MSE:
    - use_attention_transfer: Enable temporal attention matching
    - use_layer_norm: Apply LayerNorm before MSE (safer cross-architecture)
    - at_weight: Relative weight of AT loss vs MSE loss

    Usage:
        student_outputs = {
            'blocks': List[12] of [B, T, D],
            'final': [B, T, D]
        }
        teacher_outputs = {
            'blocks': List[32] of [B, T, D_teacher],  # All teacher blocks
            'final': [B, T, D_teacher]
        }
        # With teacher_layer_indices=[2,5,7,10,13,15,18,21,23,26,29,31]
        # -> maps student[0]->teacher[2], student[1]->teacher[5], etc.
    """

    def __init__(self, loss_type='mse', delta=1.0,
                 block_weights=None, final_weight=1.0,
                 teacher_layer_indices=None, num_student_layers=None,
                 use_attention_transfer=False, use_layer_norm=False,
                 at_weight=1.0, mse_weight=1.0, cosine_weight=0.0,
                 use_rkd=False, rkd_weight=0.35, rkd_n_samples=64,
                 rkd_from_block=0, rkd_to_block=None, feature_norm='token'):
        """
        Args:
            block_weights: List of weights for each block (default: auto-generated)
            final_weight: Weight for final layer (default: 1.0)
            teacher_layer_indices: List of teacher block indices to use for each student block.
                                   If None, uses sequential mapping (backward compatible).
                                   Example: [2, 5, 7, 10, 13, 15, 18, 21, 23, 26, 29, 31]
            num_student_layers: Number of student layers (used to auto-generate block_weights if None)
            use_attention_transfer: Enable Attention Transfer loss (default: False)
            use_layer_norm: Apply LayerNorm to features before MSE (default: False)
            at_weight: Weight for AT loss relative to block weight (default: 1.0)
            mse_weight: Weight for MSE loss relative to block weight (default: 1.0)
            cosine_weight: Weight for cosine similarity loss (default: 0.0)
            use_rkd: Enable Relational KD loss — matches pairwise distances and angles
                     between frames instead of forcing exact attention map alignment.
                     Allows the student to develop its own attention while preserving
                     speech unit relationships. (default: False)
            rkd_weight: Weight for RKD loss relative to block weight (default: 0.35)
            rkd_n_samples: Frames to subsample per step for O(n²) efficiency (default: 64)
            rkd_from_block: First block index (inclusive) to apply RKD. (default: 0)
            rkd_to_block: Last block index (exclusive) to apply RKD. None = all blocks from rkd_from_block.
                          E.g. rkd_from_block=3, rkd_to_block=5 → RKD on blocks 3 and 4 only.
        """
        super().__init__(loss_type, delta)

        self.teacher_layer_indices = teacher_layer_indices
        self.final_weight = final_weight
        self.use_attention_transfer = use_attention_transfer
        self.use_layer_norm = use_layer_norm
        # 'token' = per-token LayerNorm across D (default, original behavior).
        # 'channel' = per-channel standardization over valid tokens — neutralizes Whisper's
        #             upper-layer "massive activation" outlier dims that per-token LN leaves
        #             ~80x over-weighted in the MSE, so the loss supervises content channels.
        self.feature_norm = feature_norm
        self.at_weight = at_weight
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.use_rkd = use_rkd
        self.rkd_weight = rkd_weight
        self.rkd_n_samples = rkd_n_samples
        self.rkd_from_block = rkd_from_block
        self.rkd_to_block = rkd_to_block  # None = no upper bound

        # Auto-generate block weights if not provided
        if block_weights is not None:
            self.block_weights = block_weights
        elif num_student_layers is not None:
            # Linearly increasing weights from 0.3 to 0.8
            self.block_weights = [
                0.3 + 0.5 * (i / (num_student_layers - 1)) if num_student_layers > 1 else 0.5
                for i in range(num_student_layers)
            ]
        else:
            self.block_weights = [0.5, 0.6, 0.7, 0.8]  # Legacy default for 4 layers

        # LayerNorm for normalized MSE (no learnable params)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=1, elementwise_affine=False)

        print(f"Distillation Strategy: Multi-Layer")
        print(f"  Block weights: {self.block_weights}")
        print(f"  Final weight: {self.final_weight}")
        rkd_range = f"{rkd_from_block}-{rkd_to_block - 1}" if rkd_to_block is not None else f"{rkd_from_block}+"
        print(f"  Attention Transfer: {use_attention_transfer} (weight={at_weight}, blocks 0-{rkd_from_block - 1})")
        print(f"  Relational KD (RKD): {use_rkd} (weight={rkd_weight}, n_samples={rkd_n_samples}, blocks {rkd_range})")
        print(f"  Layer Norm MSE: {use_layer_norm} (weight={mse_weight})")
        print(f"  Cosine Similarity: {cosine_weight > 0} (weight={cosine_weight})")
        if teacher_layer_indices:
            print(f"  Teacher layer mapping: {teacher_layer_indices}")
        else:
            print(f"  Teacher layer mapping: Sequential (no mapping)")

    def compute_rkd_loss(self, student_feat, teacher_feat, padding_mask):
        """
        Relational Knowledge Distillation (Park et al. 2019).

        Instead of forcing exact attention map alignment (AT), RKD matches the
        geometric relationships between frames: pairwise distances and angles.
        This allows the student to develop its own laser-adapted attention patterns
        while preserving the relational structure of speech representations.

        Distance-wise: normalize pairwise L2 distances, match with Huber loss.
        Angle-wise: match cosine similarity matrix (inner products of normalized frames).

        Frames are subsampled to rkd_n_samples for O(n²) efficiency (vs O(T²) for T=1500).
        """
        B, T, D = student_feat.shape

        # Subsample frames for efficiency
        n = min(self.rkd_n_samples, T)
        if T > n:
            idx = torch.randperm(T, device=student_feat.device)[:n]
            s = student_feat[:, idx, :]  # [B, n, D]
            t = teacher_feat[:, idx, :]
        else:
            s = student_feat
            t = teacher_feat

        # ── Distance-wise RKD ─────────────────────────────────────────────
        # Pairwise L2 distances: [B, n, n]
        s_dist = torch.cdist(s, s, p=2)
        t_dist = torch.cdist(t, t, p=2)
        # Normalize by mean distance (scale-invariant, as in Park et al.)
        s_dist = s_dist / (s_dist.mean() + 1e-6)
        t_dist = t_dist / (t_dist.mean() + 1e-6)
        dist_loss = F.huber_loss(s_dist, t_dist.detach(), delta=1.0)

        # ── Angle-wise RKD ────────────────────────────────────────────────
        # Cosine similarity matrix [B, n, n] — captures pairwise angular relationships
        s_norm = F.normalize(s, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)
        s_angle = torch.bmm(s_norm, s_norm.transpose(1, 2))  # [B, n, n]
        t_angle = torch.bmm(t_norm, t_norm.transpose(1, 2))
        angle_loss = F.mse_loss(s_angle, t_angle.detach())

        return dist_loss + angle_loss

    def compute_at_loss(self, student_h, teacher_h, mask, eps=1e-6):
        """
        Compute Attention Transfer loss between student and teacher hidden states.
        Matches L2-normalized temporal attention maps.

        Args:
            student_h: [B, T, D_student]
            teacher_h: [B, T, D_teacher]
            mask: [B, T] padding mask (1=valid, 0=pad)
        """
        mask = mask.to(student_h.dtype)

        # Temporal attention: sum of |activations| over feature dim (p=1)
        s_attn = student_h.abs().sum(dim=-1)  # [B, T]
        t_attn = teacher_h.abs().sum(dim=-1)  # [B, T]

        # Zero padding BEFORE normalizing
        s_attn = s_attn * mask
        t_attn = t_attn * mask

        # L2 normalize per sample
        s_attn = s_attn / (s_attn.norm(p=2, dim=-1, keepdim=True) + eps)
        t_attn = t_attn / (t_attn.norm(p=2, dim=-1, keepdim=True) + eps)

        # Masked L2 distance
        diff2 = ((s_attn - t_attn) ** 2) * mask

        # Sum over frames (not mean!) to match original AT paper (Zagoruyko 2017)
        # β scaling is handled externally via at_weight config
        valid = (mask.sum(dim=-1) > 0).to(student_h.dtype)
        loss_per_sample = diff2.sum(dim=-1)  # squared L2 distance, bounded [0, 4]
        return (loss_per_sample * valid).sum() / (valid.sum() + eps)

    def compute_cosine_loss(self, student_feat, teacher_feat, padding_mask, eps=1e-6):
        """
        Compute frame-wise cosine similarity loss: 1 - cos_sim(student, teacher).
        Captures directional alignment independent of magnitude.

        Args:
            student_feat: [B, T, D_student]
            teacher_feat: [B, T, D_teacher]
            padding_mask: [B, T]
        """
        # Frame-wise cosine similarity: [B, T]
        cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
        loss = 1.0 - cos_sim  # [B, T], range [0, 2]

        masked_loss = loss * padding_mask
        return masked_loss.sum() / (padding_mask.sum() + eps)

    def compute_normalized_mse_loss(self, student_feat, teacher_feat, padding_mask):
        """
        Compute MSE loss with LayerNorm applied to both sides.
        More robust for cross-architecture distillation.

        Args:
            student_feat: [B, T, D_student]
            teacher_feat: [B, T, D_teacher]
            padding_mask: [B, T]
        """
        if getattr(self, 'feature_norm', 'token') == 'channel':
            # Per-channel standardization over valid (B*T) tokens, computed in float32.
            # Each of the D channels is z-scored using stats over the valid frames, so a
            # single huge-magnitude outlier channel no longer dominates the MSE; every
            # channel contributes equally and the loss supervises content, not the outlier.
            s = student_feat.float()
            t = teacher_feat.float()
            m = padding_mask.bool().unsqueeze(-1)  # [B, T, 1]

            def _chan_std(x):
                valid = x.masked_select(m.expand_as(x)).view(-1, x.size(-1))  # [Nvalid, D]
                mu = valid.mean(0)
                sd = valid.std(0) + 1e-6
                return (x - mu) / sd

            s_norm = _chan_std(s).to(student_feat.dtype)
            t_norm = _chan_std(t).to(teacher_feat.dtype)
        else:
            # 'token': per-token LayerNorm along feature dim (float32 to avoid BF16 overflow)
            s_norm = F.layer_norm(student_feat.float(), [student_feat.size(-1)]).to(student_feat.dtype)
            t_norm = F.layer_norm(teacher_feat.float(), [teacher_feat.size(-1)]).to(teacher_feat.dtype)

        # Standard masked MSE — dim-weighted collapse over D when configured
        loss = self.loss_fn(s_norm, t_norm)  # [B, T, D]
        loss = self._collapse_over_dim(loss)  # [B, T]
        masked_loss = loss * padding_mask
        return masked_loss.sum() / (padding_mask.sum() + 1e-6)

    def forward(self, student_outputs, teacher_outputs, padding_mask, temporal_window=None):
        """
        Args:
            student_outputs: Dict with 'blocks' (List[Tensor]) and 'final' (Tensor)
            teacher_outputs: Dict with 'blocks' (List[Tensor]) and 'final' (Tensor)
            padding_mask: [B, T]
            temporal_window: int or None. If set, randomly sample this many contiguous
                             tokens for distillation so all temporal positions receive
                             equal gradient pressure over training.

        Returns:
            Dict with losses:
            - total: Combined weighted loss
            - block_i_mse: MSE loss for block i
            - block_i_at: AT loss for block i (if enabled)
            - final_mse, final_at: Final layer losses
            - at_total, mse_total: Aggregated AT and MSE losses
        """
        total_loss = 0
        losses = {}
        total_mse = 0
        total_at = 0
        total_cosine = 0

        # Random temporal window: gives equal gradient pressure to all token positions.
        # Without this, early tokens dominate because they are easier to align and
        # the model stops improving late-sequence representations (visible as hallucinations
        # in the second half of 30s laser utterances).
        if temporal_window is not None:
            T = padding_mask.shape[1]
            if temporal_window < T:
                import torch
                t_start = torch.randint(0, T - temporal_window + 1, (1,)).item()
                t_end = t_start + temporal_window
                padding_mask = padding_mask[:, t_start:t_end]
                # Slice all feature tensors in student/teacher dicts
                def _slice(d):
                    return {k: ([f[:, t_start:t_end] for f in v] if isinstance(v, list)
                                else v[:, t_start:t_end])
                            for k, v in d.items()}
                student_outputs = _slice(student_outputs)
                teacher_outputs = _slice(teacher_outputs)

        # Block losses
        student_blocks = student_outputs['blocks']
        teacher_blocks = teacher_outputs['blocks']

        # Apply layer mapping if specified AND teacher has more blocks than needed
        # (If teacher_blocks already has exact count, they're pre-filtered from cache)
        if self.teacher_layer_indices is not None and len(teacher_blocks) > len(self.teacher_layer_indices):
            # Full teacher blocks available - select specific ones
            mapped_teacher_blocks = [teacher_blocks[idx] for idx in self.teacher_layer_indices]
        else:
            # Sequential mapping: either no mapping needed, or blocks are pre-filtered from cache
            mapped_teacher_blocks = teacher_blocks

        # Ensure we have enough weights for all student blocks
        num_student_blocks = len(student_blocks)
        if len(self.block_weights) < num_student_blocks:
            # Extend weights with the last weight value
            self.block_weights = self.block_weights + [self.block_weights[-1]] * (num_student_blocks - len(self.block_weights))

        for i, (s_block, t_block) in enumerate(zip(student_blocks, mapped_teacher_blocks)):
            weight = self.block_weights[i] if i < len(self.block_weights) else 0.5

            # MSE loss (optionally with LayerNorm)
            if self.use_layer_norm:
                mse_loss = self.compute_normalized_mse_loss(s_block, t_block, padding_mask)
            else:
                mse_loss = self.compute_masked_loss(s_block, t_block, padding_mask)

            losses[f'block_{i}_mse'] = mse_loss
            total_mse += mse_loss * weight
            total_loss += mse_loss * weight * self.mse_weight

            # Attention Transfer (early blocks) vs Relational KD (deep blocks)
            # rkd_from_block controls the split: AT for i < rkd_from_block, RKD for i >= rkd_from_block
            if self.use_attention_transfer and i < self.rkd_from_block:
                at_loss = self.compute_at_loss(s_block, t_block, padding_mask)
                losses[f'block_{i}_at'] = at_loss
                total_at += at_loss * weight
                total_loss += at_loss * weight * self.at_weight

            if self.use_rkd and i >= self.rkd_from_block and (self.rkd_to_block is None or i < self.rkd_to_block):
                rkd_loss = self.compute_rkd_loss(s_block, t_block, padding_mask)
                losses[f'block_{i}_rkd'] = rkd_loss
                total_loss += rkd_loss * weight * self.rkd_weight

            # Cosine similarity loss (if enabled)
            if self.cosine_weight > 0:
                cosine_loss = self.compute_cosine_loss(s_block, t_block, padding_mask)
                losses[f'block_{i}_cosine'] = cosine_loss
                total_cosine += cosine_loss * weight
                total_loss += cosine_loss * weight * self.cosine_weight

        # Final layer losses
        s_final = student_outputs['final']
        t_final = teacher_outputs['final']

        if self.use_layer_norm:
            final_mse = self.compute_normalized_mse_loss(s_final, t_final, padding_mask)
        else:
            final_mse = self.compute_masked_loss(s_final, t_final, padding_mask)

        losses['final_mse'] = final_mse
        total_mse += final_mse * self.final_weight
        total_loss += final_mse * self.final_weight * self.mse_weight

        if self.use_attention_transfer:
            final_at = self.compute_at_loss(s_final, t_final, padding_mask)
            losses['final_at'] = final_at
            total_at += final_at * self.final_weight
            total_loss += final_at * self.final_weight * self.at_weight

        if self.use_rkd:
            final_rkd = self.compute_rkd_loss(s_final, t_final, padding_mask)
            losses['final_rkd'] = final_rkd
            total_loss += final_rkd * self.final_weight * self.rkd_weight

        if self.cosine_weight > 0:
            final_cosine = self.compute_cosine_loss(s_final, t_final, padding_mask)
            losses['final_cosine'] = final_cosine
            total_cosine += final_cosine * self.final_weight
            total_loss += final_cosine * self.final_weight * self.cosine_weight

        # Aggregate losses for logging
        losses['mse_total'] = total_mse
        if self.use_attention_transfer:
            losses['at_total'] = total_at
        if self.cosine_weight > 0:
            losses['cosine_total'] = total_cosine
        losses['total'] = total_loss

        return losses


class ComponentWiseDistillation(DistillationStrategy):
    """
    Component-wise: Distill specific architectural components

    Components:
    - stem: Preprocessing outputs (Whisper Conv2)
    - attention: Attention outputs within blocks
    - blocks: Full block outputs
    - final: Final encoder output

    Usage:
        student_outputs = {
            'stem': [B, T, D],                     # Optional
            'attention': List[4] of [B, T, D],     # Optional
            'conv': List[4] of [B, T, D],          # Optional
            'blocks': List[4] of [B, T, D],        # Optional
            'final': [B, T, D]
        }
    """

    def __init__(self, loss_type='mse', delta=1.0,
                 component_weights=None):
        """
        Args:
            component_weights: Dict of {component: weight}
                Default: {
                    'stem': 0.5,
                    'attention': 0.3,
                    'conv': 0.5,
                    'blocks': 0.8,
                    'final': 2.0
                }
        """
        super().__init__(loss_type, delta)

        self.component_weights = component_weights or {
            'stem': 0.5,
            'attention': 0.3,
            'conv': 0.5,
            'blocks': 0.8,
            'final': 2.0
        }

        print(f"Distillation Strategy: Component-Wise")
        print(f"  Component weights: {self.component_weights}")

    def forward(self, student_outputs, teacher_outputs, padding_mask):
        """
        Args:
            student_outputs: Dict with optional keys: 'stem', 'attention', 'conv', 'blocks', 'final'
            teacher_outputs: Dict with same keys
            padding_mask: [B, T]
        """
        total_loss = 0
        losses = {}

        # Stem loss
        if 'stem' in student_outputs and 'stem' in teacher_outputs:
            stem_loss = self.compute_masked_loss(
                student_outputs['stem'],
                teacher_outputs['stem'],
                padding_mask
            )
            weight = self.component_weights.get('stem', 0.5)
            total_loss += stem_loss * weight
            losses['stem'] = stem_loss

        # Attention losses (per block)
        if 'attention' in student_outputs and 'attention' in teacher_outputs:
            attn_weight = self.component_weights.get('attention', 0.3)
            attn_losses = []

            for i, (s_attn, t_attn) in enumerate(
                zip(student_outputs['attention'], teacher_outputs['attention'])
            ):
                attn_loss = self.compute_masked_loss(s_attn, t_attn, padding_mask)
                total_loss += attn_loss * attn_weight
                attn_losses.append(attn_loss)
                losses[f'attention_{i}'] = attn_loss

            if attn_losses:
                losses['attention_mean'] = sum(attn_losses) / len(attn_losses)

        # Conv losses (per block)
        if 'conv' in student_outputs and 'conv' in teacher_outputs:
            conv_weight = self.component_weights.get('conv', 0.5)
            conv_losses = []

            for i, (s_conv, t_conv) in enumerate(
                zip(student_outputs['conv'], teacher_outputs['conv'])
            ):
                conv_loss = self.compute_masked_loss(s_conv, t_conv, padding_mask)
                total_loss += conv_loss * conv_weight
                conv_losses.append(conv_loss)
                losses[f'conv_{i}'] = conv_loss

            if conv_losses:
                losses['conv_mean'] = sum(conv_losses) / len(conv_losses)

        # Block losses (per block)
        if 'blocks' in student_outputs and 'blocks' in teacher_outputs:
            block_weight = self.component_weights.get('blocks', 0.8)
            block_losses = []

            for i, (s_block, t_block) in enumerate(
                zip(student_outputs['blocks'], teacher_outputs['blocks'])
            ):
                block_loss = self.compute_masked_loss(s_block, t_block, padding_mask)
                total_loss += block_loss * block_weight
                block_losses.append(block_loss)
                losses[f'block_{i}'] = block_loss

            if block_losses:
                losses['blocks_mean'] = sum(block_losses) / len(block_losses)

        # Final loss (always present)
        final_loss = self.compute_masked_loss(
            student_outputs['final'],
            teacher_outputs['final'],
            padding_mask
        )
        final_weight = self.component_weights.get('final', 2.0)
        total_loss += final_loss * final_weight
        losses['final'] = final_loss

        losses['total'] = total_loss
        return losses


def create_distillation_strategy(strategy_type, **kwargs):
    """
    Factory function to create distillation strategy

    Args:
        strategy_type: 'single', 'multi_layer', or 'component_wise'
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        DistillationStrategy instance
    """
    if strategy_type == 'single':
        return SingleLayerDistillation(**kwargs)
    elif strategy_type == 'multi_layer':
        return MultiLayerDistillation(**kwargs)
    elif strategy_type == 'component_wise':
        return ComponentWiseDistillation(**kwargs)
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}. "
                        f"Must be 'single', 'multi_layer', or 'component_wise'")


# Example usage:
if __name__ == "__main__":
    # Test single-layer
    strategy = create_distillation_strategy('single', loss_type='mse')

    student_out = {'final': torch.randn(2, 100, 256)}
    teacher_out = {'final': torch.randn(2, 100, 256)}
    mask = torch.ones(2, 100)

    losses = strategy(student_out, teacher_out, mask)
    print("Single-layer losses:", losses)

    # Test multi-layer
    strategy = create_distillation_strategy('multi_layer',
                                           block_weights=[0.5, 0.6, 0.7, 0.8],
                                           final_weight=1.0)

    student_out = {
        'blocks': [torch.randn(2, 100, 256) for _ in range(4)],
        'final': torch.randn(2, 100, 256)
    }
    teacher_out = {
        'blocks': [torch.randn(2, 100, 256) for _ in range(4)],
        'final': torch.randn(2, 100, 256)
    }

    losses = strategy(student_out, teacher_out, mask)
    print("Multi-layer losses:", losses)

    # Test component-wise
    strategy = create_distillation_strategy('component_wise',
                                           component_weights={
                                               'stem': 0.5,
                                               'blocks': 0.8,
                                               'final': 2.0
                                           })

    student_out = {
        'stem': torch.randn(2, 100, 256),
        'blocks': [torch.randn(2, 100, 256) for _ in range(4)],
        'final': torch.randn(2, 100, 256)
    }
    teacher_out = {
        'stem': torch.randn(2, 100, 256),
        'blocks': [torch.randn(2, 100, 256) for _ in range(4)],
        'final': torch.randn(2, 100, 256)
    }

    losses = strategy(student_out, teacher_out, mask)
    print("Component-wise losses:", losses)
