import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


class ResidualAdapter(nn.Module):
    """Zero-init residual adapter: x + fc2(GELU(fc1(LN(x)))).
    Starts as exact identity; learns per-dim scale correction via CE gradients.
    Used when freeze_encoder=True to provide the only trainable path to the decoder.
    """
    def __init__(self, dim, bottleneck_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class LaserWhisperStudent(nn.Module):
    """
    Student encoder using the published Whisper tiny architecture.
    Only the first Conv1d is replaced to accept laser frequency bins instead of mel bins.
    Everything else (transformer blocks, positional encoding, layer norm) is exactly as published.
    """

    def __init__(self,
                 input_freq=201,
                 student_dim=384,       # Whisper tiny: 384
                 teacher_dim=768,       # Whisper small: 768
                 num_layers=4,          # Whisper tiny: 4
                 num_heads=6,           # Whisper tiny: 6
                 whisper_model_name="tiny.en",
                 use_gradient_checkpointing=False,
                 pretrain_mode=None,
                 use_per_block_projections=False,
                 nonlinear_block_projections=False,  # 2-layer MLP per-block (False=linear, backward compat)
                 student_distill_layer_indices=None,  # Which student layers to use for distillation (None=all)
                 use_vad=False,          # DEPRECATED: input-masking VAD was removed. VAD is now a
                 vad_threshold=0.1,      # symmetric loss mask handled by system.py via batch['vad_mask'].
                 vad_smooth_frames=5,    # These args are kept for config compatibility but are ignored.
                 force_projection_head=False,  # Replace Identity with ResidualAdapter when dims match
                 adapter_bottleneck_dim=256,   # Bottleneck dim for ResidualAdapter
                 use_input_norm=True,   # Ablation flag: set False to disable InstanceNorm2d pre-stem
                 use_pre_stem=True,     # Ablation flag: set False to disable the 2D pre-stem conv
                 **kwargs):             # Accept and ignore extra args
        super().__init__()
        self.use_input_norm = use_input_norm
        self.use_pre_stem = use_pre_stem

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.pretrain_mode = pretrain_mode
        self.use_per_block_projections = use_per_block_projections
        self.num_layers = num_layers
        self.teacher_dim = teacher_dim
        self.student_distill_layer_indices = student_distill_layer_indices  # e.g. [6,7,8,9,10,11] for small
        # VAD attrs retained for config compatibility but no longer drive input masking;
        # see forward() for the removed input-masking path. VAD is now a symmetric loss
        # mask in system.py — see scripts/precompute_vad.py for the producer.
        self.use_vad = use_vad
        self.vad_threshold = vad_threshold
        self.vad_smooth_frames = vad_smooth_frames
        if use_vad:
            print(f"VAD (loss-mask mode): threshold={vad_threshold}, smooth_frames={vad_smooth_frames} — applied in system.py, not on input")

        print(f"Initializing Whisper Student Encoder from: {whisper_model_name}")
        print(f"Input frequency bins: {input_freq}")

        # Load the published Whisper encoder
        whisper_model = whisper.load_model(whisper_model_name, device="cpu")
        self.encoder = whisper_model.encoder

        # Verify dimensions match config
        n_state = self.encoder.conv1.out_channels
        assert n_state == student_dim, f"Whisper {whisper_model_name} has dim={n_state}, config says student_dim={student_dim}"
        assert len(self.encoder.blocks) == num_layers, f"Whisper {whisper_model_name} has {len(self.encoder.blocks)} blocks, config says num_layers={num_layers}"

        # Input normalization
        # Kept as a module unconditionally so legacy checkpoints load cleanly;
        # use_input_norm=False bypasses it in forward() for ablation studies.
        self.input_norm = nn.InstanceNorm2d(1, affine=True)

        # 2D pre-stem: captures frequency-time 2D structure (micro-Doppler patterns)
        # before Whisper's Conv1d layers, which treat frequency bins independently.
        # Residual connection: x = x + pre_stem_2d(x)
        # Final layer zero-initialized → starts as identity, learned incrementally.
        # Safe to load old checkpoints: missing key is ignored, module starts as ~identity.
        self.pre_stem_2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.pre_stem_2d[2].weight)
        nn.init.zeros_(self.pre_stem_2d[2].bias)
        print("2D pre-stem: residual Conv2d(1->32, k=3x7) -> GELU -> Conv2d(32->1, k=1x1), zero-init")

        # Replace conv1 only if input_freq differs from Whisper's original
        original_conv1 = self.encoder.conv1
        if input_freq != original_conv1.in_channels:
            self.encoder.conv1 = nn.Conv1d(
                in_channels=input_freq,
                out_channels=n_state,
                kernel_size=original_conv1.kernel_size[0],
                padding=original_conv1.padding[0]
            )
            print(f"Replaced conv1: {original_conv1.in_channels} -> {input_freq} input channels (random init)")
        else:
            print(f"Kept pretrained conv1: {input_freq} input channels (matches Whisper — pretrained weights preserved)")
        print(f"Encoder: {num_layers} blocks, {n_state}d, {num_heads} heads (published Whisper {whisper_model_name})")

        # Free the decoder and other parts we don't need
        del whisper_model.decoder
        del whisper_model

        # Projection head: student_dim -> teacher_dim
        if student_dim != teacher_dim:
            mid_dim = (student_dim + teacher_dim) // 2  # 384->576 or similar gradual expansion
            self.projection_head = nn.Sequential(
                nn.Linear(student_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim, teacher_dim),
                nn.LayerNorm(teacher_dim)
            )
            print(f"Projection head: [{student_dim} -> {mid_dim} -> {teacher_dim}]")
        elif force_projection_head:
            self.projection_head = ResidualAdapter(teacher_dim, adapter_bottleneck_dim)
            print(f"Projection head: ResidualAdapter [{teacher_dim} -> {adapter_bottleneck_dim} -> {teacher_dim}] (zero-init, force_projection_head=True)")
        else:
            # Same dim — use identity, no learned parameters needed
            self.projection_head = nn.Identity()
            print(f"Projection head: Identity [{student_dim} -> {teacher_dim}] (dims match)")

        # Stem projection (for component-wise distillation)
        self.stem_projection = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim)
        )

        # Per-block projections for multi-layer distillation
        if use_per_block_projections:
            if nonlinear_block_projections:
                # 2-layer MLP: better cross-modal alignment, NOT compatible with linear checkpoints
                self.block_projections = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(student_dim, teacher_dim),
                        nn.LayerNorm(teacher_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(teacher_dim, teacher_dim),
                        nn.LayerNorm(teacher_dim)
                    ) for _ in range(num_layers)
                ])
                print(f"Per-block projections: {num_layers} x 2-layer MLP [{student_dim} -> {teacher_dim} -> {teacher_dim}]")
            else:
                # Linear (default): backward compatible with old checkpoints
                self.block_projections = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(student_dim, teacher_dim),
                        nn.LayerNorm(teacher_dim)
                    ) for _ in range(num_layers)
                ])
                print(f"Per-block projections: {num_layers} x linear [{student_dim} -> {teacher_dim}]")
        else:
            self.block_projections = None

        # CTC head on projected features
        self.ctc_vocab_size = 29  # 26 letters + space + apostrophe + blank
        self.ctc_head = nn.Linear(teacher_dim, self.ctc_vocab_size)
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_head.bias.data[0] = -3.0  # Make blank less likely initially

        # Intermediate CTC head: applied to block[ctc_intermediate_layer] output.
        # Layers 0..ctc_intermediate_layer learn temporal alignment via CTC;
        # layers above are free to produce MSE-compatible representations.
        self.ctc_intermediate_layer = kwargs.get('ctc_intermediate_layer', None)
        if self.ctc_intermediate_layer is not None:
            self.ctc_intermediate_head = nn.Sequential(
                nn.LayerNorm(student_dim),
                nn.Linear(student_dim, self.ctc_vocab_size)
            )
            nn.init.normal_(self.ctc_intermediate_head[1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ctc_intermediate_head[1].bias)
            self.ctc_intermediate_head[1].bias.data[0] = -3.0
            print(f"Intermediate CTC head: block[{self.ctc_intermediate_layer}] output [{student_dim} -> {self.ctc_vocab_size}] (blocks {self.ctc_intermediate_layer+1}..{num_layers-1} free)")
        else:
            self.ctc_intermediate_head = None

    @staticmethod
    def patch_attn_map_capture(encoder, block_indices):
        """Patch specified encoder blocks so their self-attention saves w=softmax(QK^T)
        in block.attn._last_attn_weights [B, n_heads, T, T] WITHOUT detaching.
        Forces manual attention (no SDPA) so w is always available.
        Call once after model init; stays active for the model's lifetime.
        """
        import types

        def _make_patched_qkv(attn_module):
            def patched_qkv_attention(self_attn, q, k, v, mask=None):
                n_batch, n_ctx, n_state = q.shape
                scale = (n_state // self_attn.n_head) ** -0.25
                q4 = q.view(n_batch, n_ctx, self_attn.n_head, -1).permute(0, 2, 1, 3)
                k4 = k.view(n_batch, n_ctx, self_attn.n_head, -1).permute(0, 2, 1, 3)
                v4 = v.view(n_batch, n_ctx, self_attn.n_head, -1).permute(0, 2, 1, 3)
                qk = (q4 * scale) @ (k4 * scale).transpose(-1, -2)
                if mask is not None:
                    qk = qk + mask[:n_ctx, :n_ctx]
                w = F.softmax(qk.float(), dim=-1).to(q.dtype)  # [B, heads, T, T]
                out = (w @ v4).permute(0, 2, 1, 3).flatten(start_dim=2)
                self_attn._last_attn_weights = w.mean(dim=1)  # [B, T, T] avg heads, kept in graph
                return out, qk.detach()
            return types.MethodType(patched_qkv_attention, attn_module)

        for i in block_indices:
            block = encoder.blocks[i]
            block.attn.qkv_attention = _make_patched_qkv(block.attn)
        print(f"Patched encoder blocks {block_indices} for attention map capture")

    def forward(self, x, padding_mask=None, return_ctc=False, return_intermediate=False):
        """
        Args:
            x: Input laser spectrogram [B, 1, F, T] where F=freq bins, T=3000
            padding_mask: [B, T]
            return_ctc: If True, also return CTC logits
            return_intermediate: If True, return intermediate block outputs

        Returns:
            Encoder features [B, 1500, teacher_dim] (plus intermediates/CTC if requested)
        """
        B, C, Freq, T = x.shape
        target_device = self.encoder.conv1.weight.device
        target_dtype = self.encoder.conv1.weight.dtype
        x = x.to(device=target_device, dtype=target_dtype)

        # Normalize input (skippable for no-InstanceNorm ablation)
        if self.use_input_norm:
            x = self.input_norm(x)  # [B, 1, F, T]

        # 2D pre-stem: residual — starts as identity, learns incrementally
        # (skippable for no-pre-stem ablation)
        if self.use_pre_stem:
            x = x + self.pre_stem_2d(x)  # [B, 1, F, T] -> [B, 1, F, T]

        # NOTE: input-masking VAD was removed — it broke teacher-student alignment
        # because cached teacher embeddings were computed *without* VAD applied.
        # VAD is now a symmetric loss mask handled in system.py via batch['vad_mask']
        # (precomputed from audio mel by scripts/precompute_vad.py).

        # Squeeze channel dim: [B, 1, F, T] -> [B, F, T] for Conv1d
        x = x.squeeze(1)  # [B, F, T]

        # Run through Whisper's conv layers (published architecture)
        # is_acoustic=True: skip GELU (SimWhisper-Codec acoustic mode for reconstruction)
        if getattr(self, 'is_acoustic', False):
            x = self.encoder.conv1(x)
            x = self.encoder.conv2(x)
        else:
            x = F.gelu(self.encoder.conv1(x))   # [B, n_state, T]
            x = F.gelu(self.encoder.conv2(x))   # [B, n_state, T/2=1500]
        x = x.permute(0, 2, 1)              # [B, 1500, n_state]

        # Add positional embedding (published Whisper)
        # is_acoustic=True: skip PE (SimWhisper-Codec finding: absolute PE hinders reconstruction)
        if not getattr(self, 'is_acoustic', False):
            x = (x + self.encoder.positional_embedding[:x.shape[1]]).to(x.dtype)

        # Store stem output for intermediate
        stem_output = x.clone() if return_intermediate else None

        # Handle padding mask: downsample from 3000 -> 1500
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=target_device)
            if padding_mask.shape[1] == 3000:
                padding_mask = padding_mask[:, ::2]
            if padding_mask.shape[1] != x.shape[1]:
                padding_mask = padding_mask[:, :x.shape[1]]

        # Run through Whisper's transformer blocks (published architecture)
        block_outputs = [] if return_intermediate else None
        intermediate_ctc_feat = None
        for i, block in enumerate(self.encoder.blocks):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if return_intermediate:
                block_outputs.append(x.clone())
            if (return_ctc and self.ctc_intermediate_layer is not None
                    and i == self.ctc_intermediate_layer):
                intermediate_ctc_feat = x.clone()

        # Final layer norm (published Whisper)
        x = self.encoder.ln_post(x)  # [B, 1500, n_state]

        # Project to teacher dimension
        projected_feats = self.projection_head(x)  # [B, 1500, teacher_dim]

        # Build intermediate outputs dict
        intermediate_dict = None
        if return_intermediate:
            # Optionally select a subset of student layers for distillation
            if self.student_distill_layer_indices is not None:
                selected_outputs = [block_outputs[i] for i in self.student_distill_layer_indices]
                selected_indices = self.student_distill_layer_indices
            else:
                selected_outputs = block_outputs
                selected_indices = list(range(len(block_outputs)))

            if self.block_projections is not None:
                blocks_projected = [
                    self.block_projections[i](block)
                    for i, block in zip(selected_indices, selected_outputs)
                ]
            else:
                blocks_projected = [self.projection_head(block) for block in selected_outputs]

            stem_projected = self.stem_projection(stem_output)
            intermediate_dict = {
                'stem': stem_output,
                'stem_projected': stem_projected,
                'blocks': selected_outputs,
                'blocks_projected': blocks_projected,
            }

        # CTC
        if return_ctc:
            if self.ctc_intermediate_layer is not None and intermediate_ctc_feat is not None:
                # Intermediate CTC: applied to block[ctc_intermediate_layer] output.
                # Upper layers are NOT influenced by CTC gradients.
                ctc_logits = self.ctc_intermediate_head(intermediate_ctc_feat)
            else:
                if self.pretrain_mode == "ctc_with_teacher":
                    ctc_input = projected_feats.detach()
                else:
                    ctc_input = projected_feats
                ctc_logits = self.ctc_head(ctc_input)

            if return_intermediate:
                return projected_feats, intermediate_dict, ctc_logits
            return projected_feats, ctc_logits

        if return_intermediate:
            return projected_feats, intermediate_dict
        return projected_feats
