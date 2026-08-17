import re
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import whisper
import jiwer
from whisper.normalizers import EnglishTextNormalizer
_WHISPER_NORMALIZER = EnglishTextNormalizer()
from src.models import LaserWhisperStudent, LaserKDSystem
from src.loss import CompositeLoss
from src.utils.ema import ModelEMA
from peft import LoraConfig, get_peft_model


def move_to_device(obj, device):
    """Recursively move tensors in nested dicts/lists to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


class LaserLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Check pretrain mode
        self.pretrain_mode = cfg.trainer.get('pretrain_mode', None)
        if self.pretrain_mode:
            print(f"\n{'='*80}")
            print(f"PRETRAIN MODE: {self.pretrain_mode.upper()}")
            print(f"{'='*80}\n")

        # 1. Load Teacher Model (skip if ctc_only mode)
        teacher = None
        self.teacher_type = None  # teacher type (always "whisper")

        if self.pretrain_mode != "ctc_only":
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load the frozen Whisper teacher
            print(f"Loading Teacher: Whisper {cfg.model.teacher_model}...")
            teacher = whisper.load_model(cfg.model.teacher_model, device=device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            # Optional: unfreeze the decoder for full end-to-end fine-tuning
            # (comparison against LoRA's parameter-efficient adaptation).
            if cfg.trainer.get('full_decoder_ft', False):
                for param in teacher.decoder.parameters():
                    param.requires_grad = True
                teacher.decoder.train()
                n_dec = sum(p.numel() for p in teacher.decoder.parameters() if p.requires_grad)
                print(f"FULL DECODER FT enabled — unfroze teacher.decoder ({n_dec/1e6:.1f}M params trainable)")
            self.teacher_type = "whisper"
            print(f"Whisper teacher loaded on {device}")
        else:
            print("SKIPPING teacher model (ctc_only mode)")

        # 2. Initialize Student
        student = LaserWhisperStudent(
            input_freq=cfg.model.input_freq,
            student_dim=cfg.model.student_dim,
            teacher_dim=cfg.model.teacher_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            whisper_model_name=cfg.model.get('whisper_student_model', 'small.en'),
            use_gradient_checkpointing=cfg.model.get('use_gradient_checkpointing', False),
            pretrain_mode=self.pretrain_mode,
            use_per_block_projections=cfg.model.get('use_per_block_projections', False),
            nonlinear_block_projections=cfg.model.get('nonlinear_block_projections', False),
            student_distill_layer_indices=cfg.model.get('student_distill_layer_indices', None),
            ctc_intermediate_layer=cfg.trainer.get('ctc_intermediate_layer', None),
            use_vad=cfg.model.get('use_vad', False),
            vad_threshold=cfg.model.get('vad_threshold', 0.1),
            vad_smooth_frames=cfg.model.get('vad_smooth_frames', 5),
            force_projection_head=cfg.model.get('force_projection_head', False),
            adapter_bottleneck_dim=cfg.model.get('adapter_bottleneck_dim', 256),
            use_input_norm=cfg.model.get('use_input_norm', True),
            use_pre_stem=cfg.model.get('use_pre_stem', True),
        )

        # 3. Initialize System (Student + Frozen Teacher)
        distillation_mode = cfg.trainer.get('distillation_mode', 'single')  # Default: single-layer
        skip_decoder = (cfg.trainer.get('w_kd', 0.0) == 0.0 and cfg.trainer.get('w_task', 0.0) == 0.0)  # Skip if no decoder losses
        # Decoder-guided S1: run frozen decoder for CE signal, but do NOT apply LoRA/DoRA
        self.decoder_guided_s1 = cfg.trainer.get('decoder_guided_s1', False)
        if self.decoder_guided_s1:
            print("\n>>> DECODER-GUIDED S1 MODE: Frozen decoder used for CE signal (no LoRA)")
        self.system = LaserKDSystem(
            student_model=student,
            teacher_model=teacher,
            teacher_type=self.teacher_type if self.teacher_type is not None else "whisper",
            distillation_mode=distillation_mode,
            skip_decoder=skip_decoder
        )

        self.skip_decoder = skip_decoder

        # 3.5. Apply LoRA/DoRA to Teacher Decoder (only in end-to-end mode with decoder losses)
        # Skipped in decoder_guided_s1 mode — decoder stays fully frozen, CE gradient flows into encoder only.
        # Also skipped when full_decoder_ft=true — the decoder is unfrozen for full E2E FT, no adapters.
        if (self.pretrain_mode is None
                and not skip_decoder
                and not self.decoder_guided_s1
                and not cfg.trainer.get('full_decoder_ft', False)):
            lora_rank = cfg.trainer.get('lora_rank', 34)
            use_dora = cfg.trainer.get('use_dora', False)
            lora_target_self_attn = cfg.trainer.get('lora_target_self_attn', False)
            lora_target_ffn = cfg.trainer.get('lora_target_ffn', False)
            lora_cross_qv_only = cfg.trainer.get('lora_cross_qv_only', False)
            lora_cross_kv_only = cfg.trainer.get('lora_cross_kv_only', False)
            # mmWave-Whisper (Fig. 2): LoRA on Q+V ONLY (not K/out) for self AND cross attn.
            lora_qv_only = cfg.trainer.get('lora_qv_only', False)
            lora_alpha = lora_rank * 2  # Standard convention: alpha = 2 * rank
            lora_dropout = cfg.trainer.get('lora_dropout', 0.05)

            if self.teacher_type == "whisper":
                if lora_cross_kv_only:
                    # K+V only: directly adapts how encoder features are projected into key/value
                    # space. Best match for frozen encoder — encoder output is fixed, K+V remap it.
                    whisper_targets = ["cross_attn.key", "cross_attn.value"]
                elif lora_cross_qv_only:
                    # Q+V only: adapts decoder query + value readout. mmWave-Whisper style.
                    whisper_targets = ["cross_attn.query", "cross_attn.value"]
                else:
                    whisper_targets = [
                        "cross_attn.query", "cross_attn.key", "cross_attn.value", "cross_attn.out",
                    ]
                if lora_target_self_attn:
                    whisper_targets += (["attn.query", "attn.value"] if lora_qv_only
                                        else ["attn.query", "attn.key", "attn.value", "attn.out"])
                if lora_target_ffn:
                    whisper_targets += ["mlp.0", "mlp.2"]
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=whisper_targets,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_dora=use_dora,
                    modules_to_save=None,
                )
                self.system.teacher.decoder = get_peft_model(self.system.teacher.decoder, lora_config)
                mode = "DoRA" if use_dora else "LoRA"
                print(f">>> {mode} (r={lora_rank}, self_attn={lora_target_self_attn}, ffn={lora_target_ffn}, cross_qv_only={lora_cross_qv_only}) INJECTED INTO WHISPER DECODER. Trainable Params:")
                self.system.teacher.decoder.print_trainable_parameters()

                # Optionally unfreeze decoder LayerNorms in addition to DoRA deltas.
                # Each Whisper decoder block has: attn_ln, cross_attn_ln, mlp_ln.
                # LNs recalibrate activation statistics for the new input domain (~1.5k params/block).
                # PEFT freezes everything not in target_modules; we override that here.
                if cfg.trainer.get('lora_finetune_decoder_ln', False):
                    ln_count = 0
                    for module in self.system.teacher.decoder.modules():
                        # Whisper ResidualAttentionBlock LayerNorms
                        if hasattr(module, 'attn_ln'):
                            for p in module.attn_ln.parameters():
                                p.requires_grad = True
                                ln_count += p.numel()
                        if hasattr(module, 'cross_attn_ln'):
                            for p in module.cross_attn_ln.parameters():
                                p.requires_grad = True
                                ln_count += p.numel()
                        if hasattr(module, 'mlp_ln'):
                            for p in module.mlp_ln.parameters():
                                p.requires_grad = True
                                ln_count += p.numel()
                    print(f">>> Decoder LayerNorms UNFROZEN: {ln_count:,} additional params trainable.")
        elif self.decoder_guided_s1:
            print(">>> SKIPPING LoRA/DoRA injection (decoder-guided S1 — decoder frozen, CE signal only)")
        else:
            print(">>> SKIPPING LoRA/DoRA injection (pretrain mode - no decoder training yet)")

        # 3b. (S2 option) LoRA on the STUDENT ENCODER attention layers.
        # Regularized middle-ground between freezing the encoder (no adaptation, stuck at S1)
        # and full encoder FT (drifts away from the S1-aligned representation → WER rises).
        # Freezes the encoder base weights and learns only low-rank attention deltas, so the
        # encoder refines without leaving the Stage-1 manifold. In-place injection is required
        # because LaserWhisperStudent.forward accesses encoder fields directly (self.encoder.conv1,
        # .blocks, ...) — wrapping in a PeftModel would break that access.
        # NOTE: pair with freeze_encoder=false (so the LoRA params stay trainable) and
        # use_llrd=false (LLRD param-grouping does not enumerate injected LoRA params).
        # Default off → no effect on existing runs. SMOKE-TEST with trainer.fast_dev_run=true first.
        if (self.pretrain_mode is None
                and not self.decoder_guided_s1
                and cfg.trainer.get('lora_target_encoder', False)
                and self.teacher_type == "whisper"):
            from peft import inject_adapter_in_model
            enc_rank = cfg.trainer.get('encoder_lora_rank', cfg.trainer.get('lora_rank', 32))
            # mmWave-Whisper does Q+V only on encoder self-attn; default to Q,K,V,out.
            enc_targets = (["attn.query", "attn.value"] if cfg.trainer.get('lora_qv_only', False)
                           else ["attn.query", "attn.key", "attn.value", "attn.out"])
            enc_lora_config = LoraConfig(
                r=enc_rank,
                lora_alpha=enc_rank * 2,
                target_modules=enc_targets,
                lora_dropout=cfg.trainer.get('lora_dropout', 0.05),
                bias="none",
                use_dora=cfg.trainer.get('use_dora', False),
            )
            # Freeze the encoder base first; injection re-enables grad only on LoRA params.
            for p in self.system.student.encoder.parameters():
                p.requires_grad = False
            self.system.student.encoder = inject_adapter_in_model(
                enc_lora_config, self.system.student.encoder
            )
            n_enc_lora = sum(p.numel() for p in self.system.student.encoder.parameters()
                             if p.requires_grad)
            mode = "DoRA" if cfg.trainer.get('use_dora', False) else "LoRA"
            print(f">>> {mode} INJECTED INTO STUDENT ENCODER (attn q/k/v/out, r={enc_rank}): "
                  f"{n_enc_lora:,} trainable encoder params (base frozen)")

        # 4. Initialize Loss (with KD warmup support and CTC)
        self.criterion = CompositeLoss(
            w_feat=cfg.trainer.w_feat,
            w_kd=cfg.trainer.w_kd,
            w_task=cfg.trainer.w_task,
            w_contra=cfg.trainer.get('w_contra', 0.0),
            w_ctc=cfg.trainer.get('w_ctc', 0.0),  # CTC weight for alignment learning
            kd_warmup_start=cfg.trainer.get('kd_warmup_start', cfg.trainer.w_kd),
            kd_warmup_steps=cfg.trainer.get('kd_warmup_steps', 0),
            ctc_warmup_start=cfg.trainer.get('ctc_warmup_start', cfg.trainer.get('w_ctc', 0.0)),
            ctc_warmup_steps=cfg.trainer.get('ctc_warmup_steps', 0),
            task_warmup_start=cfg.trainer.get('task_warmup_start', cfg.trainer.w_task),
            task_warmup_steps=cfg.trainer.get('task_warmup_steps', 0),
            feat_loss_type=cfg.trainer.get('feat_loss_type', 'mse'),  # MSE or Huber loss
            feat_huber_delta=cfg.trainer.get('feat_huber_delta', 1.0),  # Huber delta parameter
            ctc_layer2_weight=cfg.trainer.get('ctc_layer2_weight', 0.3),  # Layer 2 CTC weight
            ctc_layer4_weight=cfg.trainer.get('ctc_layer4_weight', 0.7),  # Layer 4 CTC weight
            distillation_mode=distillation_mode,  # "single", "multi_layer", or "component_wise"
            block_weights=cfg.trainer.get('block_weights', None),  # Per-block weights for multi-layer
            component_weights=cfg.trainer.get('component_weights', None),  # Component weights for component-wise
            teacher_layer_indices=cfg.model.get('teacher_layer_indices', None),  # Layer mapping for multi-layer
            num_student_layers=cfg.model.get('num_layers', None),  # For auto-generating block weights
            use_attention_transfer=cfg.trainer.get('use_attention_transfer', False),  # AT loss
            use_layer_norm=cfg.trainer.get('use_layer_norm', False),  # Normalized MSE
            at_weight=cfg.trainer.get('at_weight', 1.0),  # AT loss weight
            mse_weight=cfg.trainer.get('mse_weight', 1.0),  # MSE loss weight
            cosine_weight=cfg.trainer.get('cosine_weight', 0.0),  # Cosine similarity loss weight
            use_rkd=cfg.trainer.get('use_rkd', False),  # Relational KD loss
            rkd_weight=cfg.trainer.get('rkd_weight', 0.35),  # RKD loss weight
            rkd_n_samples=cfg.trainer.get('rkd_n_samples', 64),  # Frames to subsample for RKD
            rkd_from_block=cfg.trainer.get('rkd_from_block', 0),  # Hybrid: AT before this, RKD from here
            rkd_to_block=cfg.trainer.get('rkd_to_block', None),   # Exclusive upper bound for RKD (None = no limit)
            kd_temperature=cfg.trainer.get('kd_temperature', 2.0),  # KD logit temperature
            distill_temporal_window=cfg.trainer.get('distill_temporal_window', None),  # Random temporal window for distillation
            feature_norm=cfg.trainer.get('feature_norm', 'token'),  # 'token' (per-token LN) | 'channel' (per-channel std, outlier-robust)
        )
        self.distillation_mode = distillation_mode  # Store for reference

        # Per-output-dim MSE weighting (Phase A.5): load calibration from cfg path,
        # push into the distillation strategy so feat / multi-layer MSE collapse
        # over D with a weighted sum instead of uniform mean. Skips silently when
        # no path is configured or the strategy was not built.
        dim_weights_path = cfg.trainer.get('dim_weights_path', None)
        if dim_weights_path is not None:
            import numpy as _np
            import torch as _torch
            dw = _torch.from_numpy(_np.load(dim_weights_path)).float()
            strategy = getattr(self.criterion, 'distillation_strategy', None)
            if strategy is not None:
                strategy.set_dim_weights(dw)
                print(f"[dim_weights] Loaded {dw.shape} from {dim_weights_path}  "
                      f"min={dw.min().item():.3f}  mean={dw.mean().item():.3f}  max={dw.max().item():.3f}")
            else:
                print(f"[dim_weights] WARN: distillation_strategy is None — dim_weights ignored")

        print(f"KD Warmup: {self.criterion.kd_warmup_start:.2f} -> {self.criterion.kd_target:.2f} over {self.criterion.kd_warmup_steps} steps")
        if self.criterion.task_warmup_steps > 0:
            print(f"Task (CE) Warmup: {self.criterion.task_warmup_start:.2f} -> {self.criterion.task_target:.2f} over {self.criterion.task_warmup_steps} steps")
        if self.criterion.w_ctc > 0:
            print(f"CTC Loss enabled with weight={self.criterion.w_ctc} for alignment learning")
            print(f"CTC Warmup: {self.criterion.ctc_warmup_start:.4f} -> {self.criterion.ctc_target:.4f} over {self.criterion.ctc_warmup_steps} steps")
        
        # Track training steps for warmup
        self.training_steps = 0

        # Stage 2: Load pre-trained encoder weights from a stage 1 (encoder-only) checkpoint
        encoder_ckpt_path = cfg.trainer.get('encoder_ckpt_path', None)
        if encoder_ckpt_path:
            print(f"\n{'='*80}")
            print(f"STAGE 2: Loading pre-trained encoder from: {encoder_ckpt_path}")
            ckpt = torch.load(encoder_ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt['state_dict']
            # Extract only student encoder weights
            student_keys = {k: v for k, v in state_dict.items() if k.startswith('system.student.')}
            missing, unexpected = self.load_state_dict(student_keys, strict=False)
            print(f"  Loaded {len(student_keys)} student encoder parameters")
            print(f"  Missing keys (expected - decoder/loss): {len(missing)}")
            print(f"{'='*80}\n")

        # 4. Initialize EMA — AFTER encoder weights loaded so EMA starts from Stage 1 weights
        self.use_ema = cfg.trainer.get("use_ema", False)
        if self.use_ema:
            self.ema = ModelEMA(
                self.system.student,
                decay=cfg.trainer.get("ema_decay", 0.999)
            )
            print(f"EMA enabled with decay={cfg.trainer.get('ema_decay', 0.999)}")

        # Cross-attention distillation weight (MSE on attended values — weak)
        self.w_xattn = cfg.trainer.get('w_xattn', 0.0)
        if self.w_xattn > 0:
            print(f"Cross-attention distillation (wv MSE): w_xattn={self.w_xattn}")

        # Decoder cross-attention KL: KL(student || teacher) on softmax(QK^T) maps [B, T_dec, T_enc]
        # More principled than wv MSE — directly supervises WHICH encoder positions decoder attends to.
        self.w_xattn_kl = cfg.trainer.get('w_xattn_kl', 0.0)
        self.xattn_kl_blocks = cfg.trainer.get('xattn_kl_blocks', [8, 9, 10, 11])
        if self.w_xattn_kl > 0 and not self.decoder_guided_s1 and not self.skip_decoder:
            self._patch_decoder_xattn_capture()
            print(f"Decoder cross-attn KL: w_xattn_kl={self.w_xattn_kl}, blocks={self.xattn_kl_blocks}")

        self.w_enc_attn = cfg.trainer.get('w_enc_attn', 0.0)
        self.enc_attn_blocks = cfg.trainer.get('enc_attn_blocks', [8, 9, 10, 11])
        if self.w_enc_attn > 0:
            LaserWhisperStudent.patch_attn_map_capture(
                self.system.student.encoder, self.enc_attn_blocks
            )
            LaserWhisperStudent.patch_attn_map_capture(
                self.system.teacher.encoder, self.enc_attn_blocks
            )
            print(f"Encoder attention distillation: w_enc_attn={self.w_enc_attn}, blocks={self.enc_attn_blocks}")

        # Store validation outputs for WER calculation (tokenizer removed to fix DataLoader hang)
        self.validation_step_outputs = []

    @torch.no_grad()
    def _log_embedding_metrics(self, prefix, student_feats, teacher_feats, mask,
                               student_intermediate=None, teacher_intermediate=None):
        """Log cosine similarity between student and teacher embeddings."""
        # Final layer cosine similarity
        if teacher_feats is None:
            return

        # Handle multi-layer teacher_feats (dict with 'final' key)
        t_final = teacher_feats['final'] if isinstance(teacher_feats, dict) else teacher_feats

        # Masked cosine similarity: [B, T, D] -> scalar
        if mask is not None:
            m = mask.unsqueeze(-1).float()  # [B, T, 1]
            s = student_feats * m
            t = t_final * m
        else:
            s, t = student_feats, t_final

        cos = F.cosine_similarity(s, t, dim=-1)  # [B, T]
        if mask is not None:
            cos_mean = (cos * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            cos_mean = cos.mean()

        self.log(f"{prefix}/cosine_sim", cos_mean, on_step=False, on_epoch=True, sync_dist=True)

        # MSE between student and teacher final-layer embeddings (diagnostic only)
        if mask is not None:
            mse = ((s - t) ** 2).mean(dim=-1)  # [B, T]
            mse_mean = (mse * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            mse_mean = F.mse_loss(student_feats, t_final)
        self.log(f"{prefix}/mse_sim", mse_mean, on_step=False, on_epoch=True, sync_dist=True)

        # Per-block cosine similarity for multi-layer mode
        if (student_intermediate is not None and teacher_intermediate is not None
                and isinstance(student_intermediate, dict) and isinstance(teacher_intermediate, dict)):
            s_blocks = student_intermediate.get('blocks', [])
            t_blocks = teacher_intermediate.get('blocks', [])
            for i, (sb, tb) in enumerate(zip(s_blocks, t_blocks)):
                cos_b = F.cosine_similarity(sb, tb, dim=-1)
                if mask is not None:
                    cos_b = (cos_b * mask.float()).sum() / mask.float().sum().clamp(min=1)
                else:
                    cos_b = cos_b.mean()
                self.log(f"{prefix}/cosine_sim_block{i}", cos_b, on_step=False, on_epoch=True, sync_dist=True)

    def forward(self, laser, teacher_feats, dec_in, mask, return_ctc=False):
        # In pretrain modes, bypass the full system and just use student encoder
        if self.pretrain_mode is not None:
            # Only use student encoder (+ optional teacher features for pretraining)
            if return_ctc:
                student_feats, ctc_logits = self.system.student(laser, mask, return_ctc=True)
            else:
                student_feats = self.system.student(laser, mask, return_ctc=False)
                ctc_logits = None

            # Handle different pretrain modes
            if self.pretrain_mode == "ctc_only":
                # CTC-only: No teacher features
                teacher_feats = None
            elif self.pretrain_mode in ["ctc_with_teacher", "masked_distill"]:
                # CTC+Teacher or Masked Distillation: Use precomputed teacher features
                if teacher_feats is not None:
                    teacher_feats = move_to_device(teacher_feats, student_feats.device)

            return {
                "student_feats": student_feats,
                "teacher_feats": teacher_feats,  # Pass through (precomputed or None)
                "student_logits": None,  # No decoder in pretrain
                "teacher_logits": None,  # No decoder in pretrain
                "ctc_logits": ctc_logits
            }
        else:
            # End-to-end mode: use full system
            return self.system(laser, teacher_feats, dec_in, mask, return_ctc=return_ctc)

    def setup(self, stage: str) -> None:
        # DDP cannot broadcast sparse tensors (Whisper registers `alignment_heads` as sparse).
        # Densify all sparse buffers before the DDP wrapper is created.
        for module in self.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.is_sparse:
                    module.register_buffer(name, buf.to_dense())

    def on_train_epoch_start(self):
        """Keep teacher base model in eval mode (Lightning sets all modules to train at epoch start)."""
        if self.system.teacher is not None:
            self.system.teacher.eval()
            # Re-enable train mode for LoRA modules so their dropout is active during student forward
            if not self.skip_decoder:
                for name, module in self.system.teacher.decoder.named_modules():
                    if 'lora' in name.lower():
                        module.train()

    def on_train_start(self):
        """Detect and recover from NaN-weight checkpoints before training begins."""
        nan_params = [n for n, p in self.system.student.named_parameters()
                      if not torch.isfinite(p).all()]
        if nan_params:
            print(f"\n[on_train_start] WARNING: {len(nan_params)} student params have NaN/Inf weights "
                  f"(loaded from a corrupted checkpoint). Re-loading encoder from encoder_ckpt_path...")
            encoder_ckpt_path = self.cfg.trainer.get('encoder_ckpt_path', None)
            if encoder_ckpt_path:
                ckpt = torch.load(encoder_ckpt_path, map_location='cpu', weights_only=False)
                student_keys = {k: v for k, v in ckpt['state_dict'].items()
                                if k.startswith('system.student.')}
                self.load_state_dict(student_keys, strict=False)
                print(f"  Re-loaded {len(student_keys)} clean encoder parameters.\n")
            else:
                raise RuntimeError("NaN weights detected but no encoder_ckpt_path set to recover from.")

    def on_after_backward(self):
        """Log LoRA grad norm after backward — guaranteed to have gradients populated."""
        if self.pretrain_mode is None and not self.skip_decoder:
            decoder_grad_norm = 0.0
            for name, param in self.system.teacher.decoder.named_parameters():
                if param.requires_grad and param.grad is not None and 'lora' in name.lower():
                    decoder_grad_norm += param.grad.data.norm(2).item() ** 2
            decoder_grad_norm = decoder_grad_norm ** 0.5
            self.log("train/decoder_lora_grad_norm", decoder_grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        laser = batch['laser_spec']
        teacher_feats = batch['teacher_feats']  # Precomputed embeddings!
        dec_in = batch['decoder_input']
        labels = batch['labels']
        mask = batch['padding_mask']
        # vad_mask: [B, 1500] float — voiced frames from audio mel. Defaults to all-ones
        # (no masking) when the sidecar is absent. Used as a *symmetric loss mask* on
        # encoder feature losses (MSE/AT/contrastive); CE on labels is left unmasked
        # so EOS/silence supervision is preserved.
        vad_mask = batch.get('vad_mask', None)
        audio_paths = batch.get('audio_path', None)

        # Move teacher_feats to device (handles nested dicts for multi-layer distillation)
        if teacher_feats is not None:
            teacher_feats = move_to_device(teacher_feats, self.device)

        # Update KD, CTC, and Task weights based on current training step (warmup)
        current_kd_weight = self.criterion.update_kd_weight(self.training_steps)
        current_ctc_weight = self.criterion.update_ctc_weight(self.training_steps)
        current_task_weight = self.criterion.update_task_weight(self.training_steps)

        # Forward Pass (with CTC if enabled)
        use_ctc = self.criterion.w_ctc > 0
        use_xattn = (self.w_xattn > 0 and not self.skip_decoder and not self.decoder_guided_s1
                     and self.pretrain_mode is None)
        use_xattn_kl = (self.w_xattn_kl > 0 and not self.skip_decoder and not self.decoder_guided_s1
                        and self.pretrain_mode is None)
        if use_xattn:
            xattn_hooks, xattn_captured = self._register_xattn_capture()
        if use_xattn_kl:
            # Clear history on each patched cross-attn module before forward pass
            blocks = self._get_decoder_blocks(self.system.teacher.decoder)
            for i in self.xattn_kl_blocks:
                if i < len(blocks):
                    block = blocks[i]
                    if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, '_xattn_map_history'):
                        block.cross_attn._xattn_map_history = []
        outputs = self(laser, teacher_feats, dec_in, mask, return_ctc=use_ctc)
        if use_xattn:
            for h in xattn_hooks:
                h.remove()

        # Debug: Print temporal dimensions once per epoch
        if batch_idx == 0:
            print(f"Student Time: {outputs['student_feats'].shape[1]}")  # e.g. 1500
            if outputs['teacher_feats'] is not None:
                # Handle teacher_feats being a dict (multi-layer) or tensor (single)
                if isinstance(outputs['teacher_feats'], dict):
                    print(f"Teacher Time: {outputs['teacher_feats']['final'].shape[1]} (multi-layer mode)")
                    if 'blocks' in outputs['teacher_feats']:
                        print(f"  Teacher blocks: {len(outputs['teacher_feats']['blocks'])}")
                else:
                    print(f"Teacher Time: {outputs['teacher_feats'].shape[1]}")  # e.g. 1500
            else:
                print(f"Teacher Time: N/A (CTC-only mode)")

        # Prepare CTC inputs if enabled
        ctc_logits = None
        input_lengths = None
        target_lengths = None
        if use_ctc and 'ctc_logits' in outputs:
            ctc_logits = outputs['ctc_logits']

            # Handle multi-layer CTC: extract layer 4 for length calculations
            # (both layers have the same temporal dimension)
            if isinstance(ctc_logits, tuple):
                ctc_logits_layer2, ctc_logits_layer4 = ctc_logits
                ctc_logits_for_len = ctc_logits_layer4  # Use layer 4 for shape reference
            else:
                # Backward compatibility: single CTC head
                ctc_logits_for_len = ctc_logits

            # Calculate encoder sequence lengths (non-padded)
            if mask is not None:
                # Downsample mask if needed
                encoder_mask = mask
                if encoder_mask.shape[1] == 3000:
                    encoder_mask = encoder_mask[:, ::2]
                if encoder_mask.shape[1] != ctc_logits_for_len.shape[1]:
                    encoder_mask = encoder_mask[:, :ctc_logits_for_len.shape[1]]
                input_lengths = encoder_mask.sum(dim=1).long()  # [B]
            else:
                input_lengths = torch.full((ctc_logits_for_len.shape[0],), ctc_logits_for_len.shape[1], dtype=torch.long, device=ctc_logits_for_len.device)

            # Calculate target sequence lengths
            target_lengths = (labels != -100).sum(dim=1).long()  # [B]

            # Debug: Print CTC masking info once per epoch
            if batch_idx == 0:
                is_multi_layer = isinstance(ctc_logits, tuple)
                print(f"\n[CTC Masking Debug]")
                if is_multi_layer:
                    print(f"  Multi-layer CTC enabled (layers 2 & 4)")
                    print(f"  CTC layer 2 shape: {ctc_logits_layer2.shape}")
                    print(f"  CTC layer 4 shape: {ctc_logits_layer4.shape}")
                else:
                    print(f"  Single-layer CTC")
                    print(f"  CTC logits shape: {ctc_logits.shape}")  # [B, T, vocab]
                print(f"  Input lengths (valid frames): {input_lengths.cpu().tolist()}")
                print(f"  Target lengths (text tokens): {target_lengths.cpu().tolist()}")
                print(f"  Max seq length: {ctc_logits_for_len.shape[1]}")
                print(f"  Padded frames sample 0: {ctc_logits_for_len.shape[1] - input_lengths[0].item()}")

        # Calculate Loss
        # Move mask to same device as student features
        if mask is not None:
            mask = mask.to(outputs['student_feats'].device)

        # Build encoder-loss mask: pad_mask AND vad_mask. Applies to feat/AT/contra
        # losses on both student and teacher sides (symmetric, since vad_mask came
        # from the audio that produced teacher_feats). CE loss uses labels, not this
        # mask, so silence-token supervision is preserved.
        encoder_loss_mask = mask
        if vad_mask is not None and mask is not None:
            vm = vad_mask.to(mask.device)
            if vm.shape[1] != mask.shape[1]:
                # Pad mask is sometimes 3000-frame; downsample to match.
                if mask.shape[1] == 3000 and vm.shape[1] == 1500:
                    vm = vm.repeat_interleave(2, dim=1)
                elif mask.shape[1] == 1500 and vm.shape[1] == 3000:
                    vm = vm[:, ::2]
            encoder_loss_mask = mask * vm
            if batch_idx == 0:
                pad_cov = mask.float().mean().item()
                comb_cov = encoder_loss_mask.float().mean().item()
                print(f"[VAD] pad_mask coverage={pad_cov:.3f}  combined coverage={comb_cov:.3f}")
                self.log("train/vad_coverage", comb_cov, on_epoch=True, sync_dist=True)

        # Get intermediate features for multi-layer distillation
        student_feats_intermediate = outputs.get('student_feats_intermediate', None)
        teacher_feats_intermediate = outputs.get('teacher_feats_intermediate', None)

        # Move ALL intermediate features to device for multi-layer distillation
        if student_feats_intermediate is not None:
            student_feats_intermediate = move_to_device(student_feats_intermediate, self.device)
        if teacher_feats_intermediate is not None:
            teacher_feats_intermediate = move_to_device(teacher_feats_intermediate, self.device)

        # Also move teacher_feats in case it's a dict with blocks
        teacher_feats_moved = outputs['teacher_feats']
        if teacher_feats_moved is not None:
            teacher_feats_moved = move_to_device(teacher_feats_moved, self.device)

        loss, breakdown = self.criterion(
            outputs['student_feats'], teacher_feats_moved, encoder_loss_mask,
            outputs['student_logits'], outputs['teacher_logits'],
            labels,
            ctc_logits=ctc_logits,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            student_feats_intermediate=student_feats_intermediate,
            teacher_feats_intermediate=teacher_feats_intermediate
        )

        # Cross-attention distillation: penalise divergence between student and teacher
        # cross-attention attended-value outputs (per decoder layer). The wv output is
        # differentiable; qk is detached by Whisper. Gradients flow to LoRA cross-attn params.
        if use_xattn and xattn_captured[0] and xattn_captured[1]:
            xattn_loss = self._compute_xattn_loss(xattn_captured[0], xattn_captured[1])
            xattn_loss = xattn_loss.to(loss.device)
            loss = loss + self.w_xattn * xattn_loss
            self.log("train/xattn_loss", xattn_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Decoder cross-attention KL: KL(student || teacher) on softmax(QK^T) maps.
        # More principled than wv MSE — directly supervises temporal alignment of attention.
        # Gradients flow to LoRA Q/K params (teacher call is detached via no_grad context).
        if use_xattn_kl:
            xattn_kl_loss = self._compute_xattn_kl_loss()
            xattn_kl_loss = xattn_kl_loss.to(loss.device)
            loss = loss + self.w_xattn_kl * xattn_kl_loss
            self.log("train/xattn_kl_loss", xattn_kl_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Encoder self-attention map distillation: forces student encoder to attend to the
        # same temporal positions as the teacher audio encoder, fixing spike attention.
        # Teacher encoder is run live on audio_mel; student maps captured via patched qkv_attention.
        if self.w_enc_attn > 0 and audio_paths is not None:
            teacher_maps = self._get_teacher_enc_attn_maps(audio_paths)
            enc_attn_loss = self._compute_enc_attn_loss(
                teacher_maps, self.system.student.encoder, self.enc_attn_blocks
            )
            enc_attn_loss = enc_attn_loss.to(loss.device)
            loss = loss + self.w_enc_attn * enc_attn_loss
            self.log("train/enc_attn_loss", enc_attn_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Guard against NaN/Inf loss — print breakdown to identify source
        if not torch.isfinite(loss):
            # Only print diagnostics on first NaN occurrence to avoid log spam
            if not getattr(self, '_nan_printed', False):
                self._nan_printed = True
                sf = outputs['student_feats']
                tf = outputs.get('teacher_feats')
                sl = outputs.get('student_logits')
                tl = outputs.get('teacher_logits')
                print(f"[NaN DIAG] student_feats: nan={torch.isnan(sf).any().item()} inf={torch.isinf(sf).any().item()} "
                      f"min={sf.float().min().item():.3f} max={sf.float().max().item():.3f}")
                if tf is not None:
                    tf_t = tf['final'] if isinstance(tf, dict) else tf
                    print(f"[NaN DIAG] teacher_feats: nan={torch.isnan(tf_t).any().item()} inf={torch.isinf(tf_t).any().item()} "
                          f"min={tf_t.float().min().item():.3f} max={tf_t.float().max().item():.3f}")
                if sl is not None:
                    print(f"[NaN DIAG] student_logits: nan={torch.isnan(sl).any().item()} inf={torch.isinf(sl).any().item()}")
                if tl is not None:
                    print(f"[NaN DIAG] teacher_logits: nan={torch.isnan(tl).any().item()} inf={torch.isinf(tl).any().item()}")
                print(f"[NaN DIAG] laser: nan={torch.isnan(laser).any().item()} min={laser.float().min().item():.3f} max={laser.float().max().item():.3f}")
            print(f"[Step {self.training_steps}] Non-finite loss ({loss.item():.4f}), breakdown: "
                  f"feat={breakdown['loss_feat']:.4f} kd={breakdown['loss_kd']:.4f} "
                  f"task={breakdown['loss_task']:.4f} contra={breakdown['loss_contra']:.4f}")
            return None

        # Update EMA (if enabled)
        if self.use_ema:
            self.ema.update(self.system.student)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/feat_loss", breakdown['loss_feat'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/kd_loss", breakdown['loss_kd'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/task_loss", breakdown['loss_task'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/contra_loss", breakdown['loss_contra'], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ctc_loss", breakdown['loss_ctc'], on_step=False, on_epoch=True, sync_dist=True)
        # Log individual CTC layer losses if multi-layer CTC is enabled
        if 'loss_ctc_layer2' in breakdown and breakdown['loss_ctc_layer2'].item() > 0:
            self.log("train/ctc_loss_layer2", breakdown['loss_ctc_layer2'], on_step=False, on_epoch=True, sync_dist=True)
        if 'loss_ctc_layer4' in breakdown and breakdown['loss_ctc_layer4'].item() > 0:
            self.log("train/ctc_loss_layer4", breakdown['loss_ctc_layer4'], on_step=False, on_epoch=True, sync_dist=True)

        # Log per-block distillation losses for multi-layer mode
        for key, value in breakdown.items():
            if key.startswith('loss_distill_') and isinstance(value, torch.Tensor):
                self.log(f"train/{key}", value, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/kd_weight", current_kd_weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("train/ctc_weight", current_ctc_weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("train/task_weight", current_task_weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        # Log embedding similarity metrics (cosine sim between student/teacher)
        self._log_embedding_metrics(
            "train", outputs['student_feats'], teacher_feats_moved, mask,
            student_feats_intermediate, teacher_feats_intermediate
        )

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        self.training_steps += 1

        return loss

    def validation_step(self, batch, batch_idx):
        laser = batch['laser_spec']
        teacher_feats = batch['teacher_feats']  # Precomputed embeddings!
        dec_in = batch['decoder_input']
        labels = batch['labels']
        mask = batch['padding_mask']
        vad_mask = batch.get('vad_mask', None)

        # Move teacher_feats to device (handles nested dicts for multi-layer distillation)
        if teacher_feats is not None:
            teacher_feats = move_to_device(teacher_feats, self.device)

        # Use EMA model for validation if enabled
        if self.use_ema:
            original_student = self.system.student
            self.system.student = self.ema.ema_model
            outputs = self(laser, teacher_feats, dec_in, mask)
            self.system.student = original_student
        else:
            outputs = self(laser, teacher_feats, dec_in, mask)

        # Move mask to same device as student features
        if mask is not None:
            mask = mask.to(outputs['student_feats'].device)

        # Symmetric VAD loss mask: same combination as training_step.
        encoder_loss_mask = mask
        if vad_mask is not None and mask is not None:
            vm = vad_mask.to(mask.device)
            if vm.shape[1] != mask.shape[1]:
                if mask.shape[1] == 3000 and vm.shape[1] == 1500:
                    vm = vm.repeat_interleave(2, dim=1)
                elif mask.shape[1] == 1500 and vm.shape[1] == 3000:
                    vm = vm[:, ::2]
            encoder_loss_mask = mask * vm

        # Get intermediate features for multi-layer distillation
        student_feats_intermediate = outputs.get('student_feats_intermediate', None)
        teacher_feats_intermediate = outputs.get('teacher_feats_intermediate', None)

        # Move ALL intermediate features to device for multi-layer distillation
        if student_feats_intermediate is not None:
            student_feats_intermediate = move_to_device(student_feats_intermediate, self.device)
        if teacher_feats_intermediate is not None:
            teacher_feats_intermediate = move_to_device(teacher_feats_intermediate, self.device)

        # Also move teacher_feats in case it's a dict with blocks
        teacher_feats_moved = outputs['teacher_feats']
        if teacher_feats_moved is not None:
            teacher_feats_moved = move_to_device(teacher_feats_moved, self.device)

        loss, breakdown = self.criterion(
            outputs['student_feats'], teacher_feats_moved, encoder_loss_mask,
            outputs['student_logits'], outputs['teacher_logits'],
            labels,
            student_feats_intermediate=student_feats_intermediate,
            teacher_feats_intermediate=teacher_feats_intermediate
        )

        # Log Validation metrics
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/feat_loss", breakdown['loss_feat'], on_epoch=True, sync_dist=True)
        self.log("val/kd_loss", breakdown['loss_kd'], on_epoch=True, sync_dist=True)
        self.log("val/task_loss", breakdown['loss_task'], on_epoch=True, sync_dist=True)
        self.log("val/contra_loss", breakdown['loss_contra'], on_epoch=True, sync_dist=True)

        # Log embedding similarity metrics
        self._log_embedding_metrics(
            "val", outputs['student_feats'], teacher_feats_moved, mask,
            student_feats_intermediate, teacher_feats_intermediate
        )

        # Only decode predictions in end-to-end mode (not during pretrain)
        if self.pretrain_mode is None and outputs['student_logits'] is not None:
            # Decode predictions for WER calculation
            student_logits = outputs['student_logits']
            pred_tokens = student_logits.argmax(dim=-1)  # [B, SeqLen]

            # Store outputs for epoch-end WER calculation
            self.validation_step_outputs.append({
                'labels': labels,
                'predictions': pred_tokens
            })

        return loss

    def on_validation_epoch_end(self):
        """Compute WER over the full validation set (teacher-forced proxy)."""
        if len(self.validation_step_outputs) == 0:
            return

        tokenizer = self.criterion.tokenizer
        all_gt, all_pred = [], []

        for output in self.validation_step_outputs:
            labels = output['labels']       # [B, SeqLen]
            predictions = output['predictions']  # [B, SeqLen] — argmax under teacher forcing

            for i in range(labels.shape[0]):
                # Ground truth: strip padding and special tokens
                valid = labels[i][labels[i] != -100]
                text_only = valid[valid < 50256]  # exclude special tokens
                gt = _WHISPER_NORMALIZER(tokenizer.decode(text_only.cpu().tolist()))

                # Prediction: strip special tokens then normalize
                pred_raw = tokenizer.decode(predictions[i].cpu().tolist())
                pred = _WHISPER_NORMALIZER(re.sub(r'<\|[^|]*\|>', '', pred_raw))

                if gt:  # skip empty ground truths
                    all_gt.append(gt)
                    all_pred.append(pred if pred else ' ')

        if all_gt:
            avg_wer = jiwer.wer(all_gt, all_pred)
            self.log("val/wer_tf", avg_wer, prog_bar=True, on_epoch=True, sync_dist=False)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @staticmethod
    def _get_decoder_blocks(decoder):
        """Return decoder transformer blocks regardless of PEFT wrapping."""
        if hasattr(decoder, 'base_model') and hasattr(decoder.base_model, 'model'):
            return decoder.base_model.model.blocks  # PEFT-wrapped
        return decoder.blocks  # plain decoder

    def _register_xattn_capture(self):
        """Register hooks on every cross-attn block to capture attended-value outputs.

        The decoder is called TWICE per step: teacher first (inside no_grad), then student.
        Hooks capture in order, so captured[0]=teacher, captured[1]=student.
        Returns (hooks_list, captured_list) where captured[run][layer] = wv tensor.
        The wv output (out[0] of MultiHeadAttention) is differentiable — Whisper detaches
        qk (out[1]) but wv stays in the computation graph, so gradients flow to LoRA params.
        """
        decoder = self.system.teacher.decoder
        blocks = self._get_decoder_blocks(decoder)
        captured = [{}, {}]  # [teacher_run, student_run]
        run_counts = {}
        hooks = []

        for i, block in enumerate(blocks):
            if not (hasattr(block, 'cross_attn') and block.cross_attn is not None):
                continue
            run_counts[i] = 0

            def make_hook(idx):
                def hook(module, inp, out):
                    run_idx = run_counts.get(idx, 0)
                    if run_idx < 2 and isinstance(out, tuple) and out[0] is not None:
                        captured[run_idx][idx] = out[0]  # wv: [B, T_dec, D]
                    run_counts[idx] = run_idx + 1
                return hook

            hooks.append(block.cross_attn.register_forward_hook(make_hook(i)))

        return hooks, captured

    @staticmethod
    def _compute_xattn_loss(teacher_xattn, student_xattn):
        """MSE between student and teacher cross-attention attended-value outputs."""
        losses = []
        for layer in student_xattn:
            if layer in teacher_xattn:
                losses.append(F.mse_loss(
                    student_xattn[layer].float(),
                    teacher_xattn[layer].float().detach(),
                ))
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def _patch_decoder_xattn_capture(self):
        """Patch cross_attn.qkv_attention on xattn_kl_blocks to save [B, T_dec, T_enc] maps.

        The decoder is called twice per step: teacher forward (inside no_grad) then student
        forward. Each call appends to _xattn_map_history on the cross_attn module:
          index 0 (teacher, no_grad) → detached tensor
          index 1 (student)          → in-graph tensor → KL gradients flow to LoRA Q/K

        Clear _xattn_map_history at the start of each training step before calling forward.
        """
        import types

        def _make_patched(attn_module):
            def patched_qkv_attention(self_attn, q, k, v, mask=None):
                n_batch, n_ctx, n_state = q.shape
                scale = (n_state // self_attn.n_head) ** -0.25
                q4 = q.view(n_batch, n_ctx, self_attn.n_head, -1).permute(0, 2, 1, 3)
                k4 = k.view(k.shape[0], k.shape[1], self_attn.n_head, -1).permute(0, 2, 1, 3)
                v4 = v.view(v.shape[0], v.shape[1], self_attn.n_head, -1).permute(0, 2, 1, 3)
                qk = (q4 * scale) @ (k4 * scale).transpose(-1, -2)  # [B, n_head, T_dec, T_enc]
                if mask is not None:
                    qk = qk + mask[:n_ctx, :n_ctx]
                w = F.softmax(qk.float(), dim=-1).to(q.dtype)  # [B, n_head, T_dec, T_enc]
                out = (w @ v4).permute(0, 2, 1, 3).flatten(start_dim=2)
                if not hasattr(self_attn, '_xattn_map_history'):
                    self_attn._xattn_map_history = []
                # teacher call is under no_grad → w is already detached; student call stays in graph
                self_attn._xattn_map_history.append(w.mean(dim=1))  # [B, T_dec, T_enc] head-avg
                return out, qk.detach()
            return types.MethodType(patched_qkv_attention, attn_module)

        blocks = self._get_decoder_blocks(self.system.teacher.decoder)
        patched = []
        for i in self.xattn_kl_blocks:
            if i < len(blocks):
                block = blocks[i]
                if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                    block.cross_attn.qkv_attention = _make_patched(block.cross_attn)
                    patched.append(i)
        print(f"Patched decoder cross-attn blocks {patched} for KL map capture")

    def _compute_xattn_kl_loss(self):
        """KL(student || teacher) on decoder cross-attention weight maps [B, T_dec, T_enc].
        Teacher maps captured during no_grad forward (detached); student maps in-graph.
        Gradients flow to LoRA Q/K/V params in the decoder cross-attention.
        """
        blocks = self._get_decoder_blocks(self.system.teacher.decoder)
        losses = []
        for i in self.xattn_kl_blocks:
            if i >= len(blocks):
                continue
            block = blocks[i]
            if not (hasattr(block, 'cross_attn') and
                    hasattr(block.cross_attn, '_xattn_map_history')):
                continue
            history = block.cross_attn._xattn_map_history
            if len(history) < 2:
                continue
            w_t = history[0].float()   # [B, T_dec, T_enc] teacher, detached
            w_s = history[1].float()   # [B, T_dec, T_enc] student, in graph
            kl = F.kl_div(
                (w_s + 1e-8).log(),
                w_t,
                reduction='sum',
            ) / (w_s.shape[0] * w_s.shape[1])  # normalize by B * T_dec → per-row KL
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def _get_teacher_enc_attn_maps(self, audio_paths):
        """Run teacher encoder on audio and return self-attention maps for enc_attn_blocks.
        Returns dict {block_idx: w [B, T, T]} averaged over heads. Teacher is frozen.
        """
        import whisper
        from src.lightning.datamodule import load_and_pad_audio

        mels = []
        for path in audio_paths:
            wav, _ = load_and_pad_audio(str(path))
            mel = whisper.log_mel_spectrogram(wav[0], n_mels=80).to(self.device)
            mels.append(mel)
        audio_mel = torch.stack(mels)  # [B, 80, 3000]

        with torch.no_grad():
            self.system.teacher.encoder(audio_mel)

        teacher_maps = {
            i: self.system.teacher.encoder.blocks[i].attn._last_attn_weights.detach()
            for i in self.enc_attn_blocks
            if hasattr(self.system.teacher.encoder.blocks[i].attn, '_last_attn_weights')
        }
        return teacher_maps

    @staticmethod
    def _compute_enc_attn_loss(teacher_maps, student_encoder, block_indices):
        """KL divergence between student and teacher encoder self-attention maps."""
        losses = []
        for i in block_indices:
            if i not in teacher_maps:
                continue
            w_t = teacher_maps[i]  # [B, T, T] detached
            if not hasattr(student_encoder.blocks[i].attn, '_last_attn_weights'):
                continue
            w_s = student_encoder.blocks[i].attn._last_attn_weights.float()  # [B, T, T] already head-avg
            # KL(w_s || w_t): row-wise distributions
            kl = F.kl_div(
                (w_s + 1e-8).log(),
                w_t.float(),
                reduction='sum',
            ) / (w_s.shape[0] * w_s.shape[1])  # normalize by B*T_q → per-row KL
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def _get_llrd_param_groups(self, student, base_lr, llrd_decay):
        """
        Layer-wise Learning Rate Decay (LLRD).

        Deeper blocks get full base_lr. Each earlier block is multiplied by
        llrd_decay, so block i gets base_lr * llrd_decay^(num_blocks - i - 1).
        Stem/conv gets the lowest LR. Projection heads always get full base_lr.
        """
        groups = []

        if hasattr(student, 'encoder'):
            enc = student.encoder
            num_blocks = len(enc.blocks)

            # Conv stem: lowest LR (includes 2D pre-stem if present)
            conv_lr = base_lr * (llrd_decay ** num_blocks)
            conv_params = (list(student.input_norm.parameters()) +
                           list(enc.conv1.parameters()) +
                           list(enc.conv2.parameters()))
            if getattr(student, 'pre_stem_2d', None) is not None:
                conv_params += list(student.pre_stem_2d.parameters())
            groups.append({'params': conv_params, 'lr': conv_lr, 'name': 'conv_stem'})

            # Transformer blocks: LR increases with depth
            for i, block in enumerate(enc.blocks):
                block_lr = base_lr * (llrd_decay ** (num_blocks - i - 1))
                groups.append({'params': list(block.parameters()),
                                'lr': block_lr, 'name': f'block_{i}'})

            # Post-norm + projection heads: full LR
            head_params = list(enc.ln_post.parameters()) + list(student.projection_head.parameters())
            if getattr(student, 'per_block_projections', None) is not None:
                head_params += list(student.per_block_projections.parameters())
            if getattr(student, 'ctc_head', None) is not None:
                head_params += list(student.ctc_head.parameters())
            groups.append({'params': head_params, 'lr': base_lr, 'name': 'projection'})

        else:
            # Fallback: single group
            groups = [{'params': list(student.parameters()), 'lr': base_lr, 'name': 'all'}]

        lrs = [(g['name'], g['lr']) for g in groups]
        print(f"  LLRD param groups (decay={llrd_decay}): {lrs}")
        return groups

    def configure_optimizers(self):
        # Option A: freeze encoder — only train decoder DoRA
        freeze_encoder = self.cfg.trainer.get('freeze_encoder', False)
        # Option A2: freeze transformer blocks only — train conv stem + projection head
        freeze_transformer_blocks = self.cfg.trainer.get('freeze_transformer_blocks', False)
        if freeze_encoder:
            student = self.system.student
            for param in student.parameters():
                param.requires_grad = False
            # Keep output heads trainable so adapter/VAD experiments still have params to optimize
            trainable_modules = []
            if hasattr(student, 'projection_head') and not isinstance(student.projection_head, torch.nn.Identity):
                trainable_modules.append(student.projection_head)
            if hasattr(student, 'ctc_head'):
                trainable_modules.append(student.ctc_head)
            for m in trainable_modules:
                for param in m.parameters():
                    param.requires_grad = True
            student_params = [p for p in student.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in student_params)
            print(f"[configure_optimizers] Encoder FROZEN — heads trainable: {n_trainable:,} params.")
        elif freeze_transformer_blocks:
            # Freeze transformer blocks (and ln_post) — only conv stem + projection head train
            student = self.system.student
            for param in student.parameters():
                param.requires_grad = False
            # Unfreeze stem: input_norm, pre_stem_2d, conv1, conv2
            stem_modules = [student.input_norm, student.encoder.conv1, student.encoder.conv2]
            if getattr(student, 'pre_stem_2d', None) is not None:
                stem_modules.append(student.pre_stem_2d)
            for module in stem_modules:
                for param in module.parameters():
                    param.requires_grad = True
            # Unfreeze projection heads
            for module in [student.projection_head]:
                for param in module.parameters():
                    param.requires_grad = True
            if getattr(student, 'stem_projection', None) is not None:
                for param in student.stem_projection.parameters():
                    param.requires_grad = True
            # Unfreeze LayerNorms in transformer blocks — cheap (2*768 params each) but
            # critical for domain adaptation: LNs recalibrate activation statistics when
            # input distribution shifts (synthesized laser -> real laser).
            for block in student.encoder.blocks:
                for ln in [block.attn_ln, block.mlp_ln]:
                    for param in ln.parameters():
                        param.requires_grad = True
            for param in student.encoder.ln_post.parameters():
                param.requires_grad = True
            student_params = [p for p in student.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in student_params)
            print(f"[configure_optimizers] STEM+LN: transformer weights frozen, LayerNorms trainable. Trainable: {n_trainable:,} params.")
        elif self.cfg.trainer.get('freeze_encoder_except_ln', False):
            # S2 variant: freeze all encoder weights except LayerNorms.
            # Attention/FFN weights stay in Whisper space; LNs recalibrate
            # activation statistics for the new input domain. Very cheap (~37k params)
            # but gives decoder DoRA a stable, domain-adapted embedding to work with.
            student = self.system.student
            for param in student.parameters():
                param.requires_grad = False
            enc = student.encoder
            for block in enc.blocks:
                for ln in [block.attn_ln, block.mlp_ln]:
                    for param in ln.parameters():
                        param.requires_grad = True
            for param in enc.ln_post.parameters():
                param.requires_grad = True
            student_params = [p for p in student.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in student_params)
            print(f"[configure_optimizers] ENCODER-LN-ONLY: all encoder weights frozen except LayerNorms. Trainable encoder: {n_trainable:,} params.")
        elif self.cfg.trainer.get('freeze_lower_n_blocks', 0) > 0:
            # Freeze encoder blocks 0..N-1 (learned low-level features, stable across domains).
            # Train blocks N..11 + projection head fully.
            # Always unfreeze LayerNorms in ALL blocks — cheap recalibration of activation stats.
            # Best for small-dataset laser FT: prevents overfitting lower layers while adapting
            # upper speech-level features and the encoder→decoder interface.
            n_freeze = self.cfg.trainer.get('freeze_lower_n_blocks', 0)
            student = self.system.student
            enc = student.encoder
            num_blocks = len(enc.blocks)
            # Freeze everything first
            for param in student.parameters():
                param.requires_grad = False
            # Unfreeze upper blocks (n_freeze..num_blocks-1)
            for i in range(n_freeze, num_blocks):
                for param in enc.blocks[i].parameters():
                    param.requires_grad = True
            # Unfreeze LayerNorms in ALL blocks (even frozen ones)
            for block in enc.blocks:
                for ln in [block.attn_ln, block.mlp_ln]:
                    for param in ln.parameters():
                        param.requires_grad = True
            # Unfreeze ln_post and projection head
            for param in enc.ln_post.parameters():
                param.requires_grad = True
            for param in student.projection_head.parameters():
                param.requires_grad = True
            if getattr(student, 'pre_stem_2d', None) is not None:
                for param in student.pre_stem_2d.parameters():
                    param.requires_grad = True
            student_params = [p for p in student.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in student_params)
            frozen_blocks = list(range(n_freeze))
            trained_blocks = list(range(n_freeze, num_blocks))
            print(f"[configure_optimizers] FREEZE-LOWER-{n_freeze}: "
                  f"blocks {frozen_blocks} FROZEN (LNs stay trainable), "
                  f"blocks {trained_blocks} + projection TRAINABLE. "
                  f"Trainable: {n_trainable:,} params.")
        else:
            student_params = list(self.system.student.parameters())

        # Only include trainable LoRA params (not frozen base decoder weights)
        # In decoder_guided_s1: decoder has no LoRA, so decoder_params will be empty (all frozen)
        if self.pretrain_mode is None and not self.skip_decoder and not self.decoder_guided_s1:
            decoder_params = [p for p in self.system.teacher.decoder.parameters() if p.requires_grad]
            decoder_trainable = sum(p.numel() for p in decoder_params)
        else:
            decoder_params = []
            decoder_trainable = 0

        # Count trainable parameters
        student_trainable = sum(p.numel() for p in student_params if p.requires_grad)

        print("="*80)
        print("OPTIMIZER CONFIGURATION:")
        print(f"  Student Encoder Trainable Params: {student_trainable:,}")
        if self.pretrain_mode is None and not self.skip_decoder:
            print(f"  Decoder LoRA Trainable Params: {decoder_trainable:,}")
        else:
            print(f"  Decoder LoRA Trainable Params: {decoder_trainable:,} (encoder-only mode - decoder not used)")
        print(f"  Total Trainable Params: {student_trainable + decoder_trainable:,}")
        print("="*80)

        # Build parameter groups: LLRD or flat, with optional differential LR for decoder
        use_llrd = self.cfg.trainer.get('use_llrd', False)
        decoder_lr = self.cfg.trainer.get('decoder_lr', None)  # Option B: separate LR for decoder DoRA
        base_lr = self.cfg.trainer.lr

        if use_llrd:
            llrd_decay = self.cfg.trainer.get('llrd_decay', 0.8)
            param_groups = self._get_llrd_param_groups(self.system.student, base_lr, llrd_decay)
            if decoder_params:
                param_groups.append({'params': decoder_params,
                                     'lr': decoder_lr if decoder_lr else base_lr, 'name': 'decoder'})
        elif decoder_lr and decoder_params and student_params:
            # Option B: differential LR — encoder lower, decoder DoRA higher
            param_groups = [
                {'params': student_params, 'lr': base_lr, 'name': 'encoder'},
                {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'},
            ]
            print(f"[configure_optimizers] Differential LR: encoder={base_lr}, decoder DoRA={decoder_lr}")
        else:
            param_groups = student_params + decoder_params

        optimizer = optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=self.cfg.trainer.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-3
        )
        
        # Warmup + Cosine Annealing
        warmup_steps = self.cfg.trainer.get("warmup_steps", 500)
        total_steps = self.cfg.trainer.max_epochs * self.trainer.estimated_stepping_batches // self.cfg.trainer.max_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss"
            }
        }
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['training_steps'] = self.training_steps
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Restore training_steps for KD/CTC warmup continuity
        # Fallback: estimate from global_step for old checkpoints without training_steps
        self.training_steps = checkpoint.get('training_steps', checkpoint.get('global_step', 0))

        # Strip keys that no longer exist in the current architecture (e.g. projection_head
        # was replaced with nn.Identity() when student_dim == teacher_dim, so old checkpoints
        # have projection_head.0/1/2 weights that would cause strict load to fail).
        state_dict = checkpoint.get('state_dict', {})
        stale_prefixes = ['system.student.projection_head.']
        stale_keys = [k for k in list(state_dict.keys())
                      if any(k.startswith(p) for p in stale_prefixes)
                      and k not in dict(self.named_parameters())]
        if stale_keys:
            print(f"[on_load_checkpoint] Dropping {len(stale_keys)} stale keys "
                  f"(architecture change): {stale_keys[:3]}")
            for k in stale_keys:
                del state_dict[k]

        if self.use_ema and 'ema_state_dict' in checkpoint:
            missing, unexpected = self.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
            if unexpected:
                print(f"[EMA load] Ignoring unexpected keys (architecture change): {unexpected[:5]}")

        # Backward compat: old checkpoints saved with LoRA on the decoder (before skip_decoder
        # was introduced) have keys like system.teacher.decoder.base_model.model.X.
        # Remap them to plain system.teacher.decoder.X and drop LoRA-specific weights.
        # ONLY do this when the current model has a plain decoder (skip_decoder=True / encoder-only).
        # When the current model also has LoRA (skip_decoder=False / Stage 2 E2E), the checkpoint
        # keys should match directly — remapping would break the load.
        if self.skip_decoder:
            state_dict = checkpoint.get('state_dict', {})
            lora_base_prefix = 'system.teacher.decoder.base_model.model.'
            plain_prefix = 'system.teacher.decoder.'
            lora_base_keys = [k for k in list(state_dict.keys()) if k.startswith(lora_base_prefix)]
            if lora_base_keys:
                print(f"[on_load_checkpoint] Remapping {len(lora_base_keys)} LoRA decoder keys "
                      f"to plain decoder keys (encoder-only mode, checkpoint had LoRA)...")
                for k in lora_base_keys:
                    suffix = k[len(lora_base_prefix):]
                    if any(s in suffix for s in ['.lora_A.', '.lora_B.', '.lora_dropout.',
                                                  'lora_A.', 'lora_B.']):
                        del state_dict[k]
                        continue
                    if '.base_layer.' in suffix:
                        suffix = suffix.replace('.base_layer.', '.')
                    state_dict[plain_prefix + suffix] = state_dict.pop(k)
                print(f"  Done. Plain decoder keys: "
                      f"{sum(1 for k in state_dict if k.startswith(plain_prefix))}")

    def load_state_dict(self, state_dict, strict=True):
        """Override to allow loading checkpoints with new architectural layers (e.g., CTC).
        
        Uses strict=False internally so that:
        - Existing weights from the checkpoint load successfully
        - New layers (like CTC heads) initialize from scratch
        """
        # Always use non-strict loading to allow architectural changes
        print(f"[load_state_dict] Loading with strict=False to allow new layers (CTC, etc.) "
              f"to initialize from scratch...")
        return super().load_state_dict(state_dict, strict=False)