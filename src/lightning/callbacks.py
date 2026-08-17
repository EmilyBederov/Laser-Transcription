import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import whisper
import matplotlib.pyplot as plt
import numpy as np
import jiwer
import re
from whisper.normalizers import EnglishTextNormalizer
_WHISPER_NORMALIZER = EnglishTextNormalizer()


def strip_special_tokens(text):
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = ' '.join(text.split())
    return text.strip()


def greedy_decode(decoder, embeddings, tokenizer, max_len=445):
    """
    Real autoregressive inference using the Whisper decoder.
    max_len=445: Whisper positional embeddings are 448; prefix is 3 tokens → 445 new tokens max.
    4-gram no-repeat suppression prevents hallucination loops.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Use Whisper's canonical SOT for small.en (English-only): [sot, notimestamps].
    sot_sequence = list(tokenizer.sot_sequence_including_notimestamps)

    tokens = torch.tensor(
        [sot_sequence] * batch_size,
        dtype=torch.long,
        device=device
    )  # [B, 3]

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = decoder(tokens, embeddings)           # [B, cur_len, vocab]
        next_token = logits[:, -1].argmax(dim=-1)      # [B]

        # Force EOT for already-done sequences
        next_token = torch.where(
            done,
            torch.full_like(next_token, tokenizer.eot),
            next_token
        )
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

        # Mark sequences that emitted EOT
        done = done | (next_token == tokenizer.eot)

        # 4-gram no-repeat: if last 4 content tokens appeared before, stop
        if tokens.shape[1] >= 7:  # 3 prefix + at least 4 content tokens
            for b in range(batch_size):
                if done[b]:
                    continue
                content = tokens[b, 3:].tolist()  # skip sot prefix
                if len(content) >= 8:
                    last4 = tuple(content[-4:])
                    prev_ngrams = [tuple(content[i:i+4]) for i in range(len(content) - 4)]
                    if last4 in prev_ngrams:
                        done[b] = True
                        tokens[b, -1] = tokenizer.eot

        if done.all():
            break

    decoded_text = []
    for t in tokens:
        text = tokenizer.decode(t.tolist())
        text = strip_special_tokens(text)
        decoded_text.append(text)

    return decoded_text


class LogPredictionsCallback(pl.Callback):
    """
    Real autoregressive inference callback.

    Every epoch:
        - Greedy-decodes num_display (8) samples → WandB table + logs val/wer.
        - val/wer excludes samples with per-sample WER > 1.0 (hallucination loops).

    Every save_every_n_epochs:
        - Greedy-decodes num_save (50) samples → CSV + logs val/wer_greedy.
    """

    def __init__(self, tokenizer, num_display=8, num_save=50,
                 save_every_n_epochs=5, output_dir=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_display = num_display
        self.num_save = num_save
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir

        self._display_batches = []   # ≤ num_display samples, every epoch
        self._save_batches = []      # ≤ num_save samples, every N epochs
        self._last_wer_greedy = None  # cached so ModelCheckpoint can monitor every epoch

    def on_validation_epoch_start(self, trainer, pl_module):
        self._display_batches = []
        self._save_batches = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        do_save = (trainer.current_epoch % self.save_every_n_epochs == 0)

        n_disp = sum(b['laser'].shape[0] for b in self._display_batches)
        n_save = sum(b['laser'].shape[0] for b in self._save_batches)

        def _slice(batch, k):
            return {
                'laser':  batch['laser_spec'][:k].cpu(),
                'labels': batch['labels'][:k].cpu(),
                'mask':   batch['padding_mask'][:k].cpu()
                          if batch.get('padding_mask') is not None else None,
            }

        bs = batch['laser_spec'].shape[0]

        if n_disp < self.num_display:
            k = min(self.num_display - n_disp, bs)
            self._display_batches.append(_slice(batch, k))

        if do_save and n_save < self.num_save:
            k = min(self.num_save - n_save, bs)
            self._save_batches.append(_slice(batch, k))

    # ------------------------------------------------------------------
    def _decode_batches(self, pl_module, batches):
        """Returns (gts, preds, laser_tensor). Real autoregressive inference."""
        all_gts, all_preds, all_laser = [], [], []
        student = pl_module.ema.ema_model if pl_module.use_ema else pl_module.system.student
        student = student.to(pl_module.device)
        decoder = pl_module.system.teacher.decoder.to(pl_module.device)

        with torch.no_grad():
            for b in batches:
                laser  = b['laser'].to(pl_module.device)
                labels = b['labels'].to(pl_module.device)
                mask   = b['mask'].to(pl_module.device) if b['mask'] is not None else None

                feats = student(laser, padding_mask=mask)
                if isinstance(feats, tuple):
                    feats = feats[0]

                if mask is not None:
                    m = mask.to(feats.device)
                    if m.shape[1] == 3000:
                        m = m[:, ::2]
                    if m.shape[1] != feats.shape[1]:
                        m = m[:, :feats.shape[1]]
                    feats = feats * m.unsqueeze(-1)

                feats = F.layer_norm(feats.float(), [feats.shape[-1]]).to(feats.dtype)
                preds = greedy_decode(decoder, feats, self.tokenizer)

                for i in range(labels.shape[0]):
                    valid = labels[i][labels[i] != -100]
                    gt = strip_special_tokens(self.tokenizer.decode(valid.cpu().tolist()))
                    all_gts.append(gt)
                all_preds.extend(preds)
                all_laser.append(laser.cpu())

        return all_gts, all_preds, torch.cat(all_laser, dim=0)

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._display_batches:
            return

        pl_module.eval()
        try:
            epoch = trainer.current_epoch
            do_save = (epoch % self.save_every_n_epochs == 0)

            # ── 1. Decode 8 samples → val/wer + WandB table (every epoch) ──
            gts, preds, laser_all = self._decode_batches(pl_module, self._display_batches)

            norm_gts   = [_WHISPER_NORMALIZER(g) for g in gts]
            norm_preds = [_WHISPER_NORMALIZER(p) for p in preds]

            # Per-sample WER; exclude WER>1 (hallucination loops) from aggregate
            per_wer = []
            for g, p in zip(norm_gts, norm_preds):
                try:
                    per_wer.append(jiwer.wer(g, p))
                except Exception:
                    per_wer.append(1.0)
            capped = [w for w in per_wer if w <= 1.0]
            wer_display = sum(capped) / len(capped) if capped else 1.0
            n_excluded = len(per_wer) - len(capped)

            pl_module.log('val/wer', wer_display, prog_bar=True,
                          on_epoch=True, sync_dist=True)
            print(f"[Callback] val/wer (autoregressive, {len(capped)}/{len(gts)} samples"
                  f"{', excl ' + str(n_excluded) + ' hallucinations' if n_excluded else ''}"
                  f") = {wer_display:.4f}")

            columns = ["Laser (Mel)", "Ground Truth", "Prediction", "WER"]
            rows = []
            for i in range(len(preds)):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(laser_all[i].cpu().numpy().squeeze(),
                          aspect='auto', origin='lower', cmap='viridis')
                ax.set_title("Laser Mel Spec")
                ax.set_xlabel("Time")
                plt.tight_layout()
                img = wandb.Image(fig)
                plt.close(fig)
                rows.append([img, gts[i], norm_preds[i], f"{per_wer[i]:.2%}"])
            try:
                trainer.logger.experiment.log(
                    {"val/predictions": wandb.Table(columns=columns, data=rows)}
                )
            except Exception as e:
                print(f"[Callback] Warning: failed to log table: {e}")

            # ── 2. Every N epochs: decode 50 samples → CSV + val/wer_greedy ──
            if do_save and self._save_batches:
                print(f"[Callback] Decoding {self.num_save} samples for CSV (epoch {epoch})...")
                all_gts, all_preds, _ = self._decode_batches(pl_module, self._save_batches)

                norm_gts2   = [_WHISPER_NORMALIZER(g) for g in all_gts]
                norm_preds2 = [_WHISPER_NORMALIZER(p) for p in all_preds]

                per_wer2 = []
                for g, p in zip(norm_gts2, norm_preds2):
                    try:
                        per_wer2.append(jiwer.wer(g, p))
                    except Exception:
                        per_wer2.append(1.0)
                capped2 = [w for w in per_wer2 if w <= 1.0]
                true_wer = sum(capped2) / len(capped2) if capped2 else 1.0
                n_exc2 = len(per_wer2) - len(capped2)
                print(f"[Callback] val/wer_greedy ({len(capped2)}/{len(all_gts)} samples"
                      f"{', excl ' + str(n_exc2) + ' hallucinations' if n_exc2 else ''}"
                      f") = {true_wer:.4f}")
                self._last_wer_greedy = true_wer
                try:
                    trainer.logger.experiment.log({"val/wer_greedy": true_wer})
                    pl_module.log("val/wer_greedy", true_wer, prog_bar=True, sync_dist=True)
                except Exception as e:
                    print(f"[Callback] Warning: failed to log wer_greedy: {e}")

                out_dir = self.output_dir or trainer.default_root_dir
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, f"val_transcriptions_epoch{epoch:03d}.csv")
                pd.DataFrame({
                    'gt':   all_gts,
                    'pred': all_preds,
                    'wer':  per_wer2,
                }).to_csv(csv_path, index=False)
                print(f"[Callback] Saved {len(all_gts)} transcriptions → {csv_path}")

        except Exception as e:
            import traceback
            print(f"[LogPredictionsCallback] ERROR: {e}")
            traceback.print_exc()
        finally:
            pl_module.train()

        # Re-log cached wer_greedy every epoch so ModelCheckpoint can always monitor it
        if self._last_wer_greedy is not None and not do_save:
            try:
                pl_module.log("val/wer_greedy", self._last_wer_greedy, prog_bar=True, sync_dist=True)
            except Exception:
                pass


class LogAttnMapsCallback(pl.Callback):
    """Log encoder self-attention maps (student vs teacher) to WandB every N epochs.
    Only active when pl_module.w_enc_attn > 0 (encoders are patched).
    Shows [T, T] head-averaged softmax(QK^T) heatmaps — useful for diagnosing
    spike/degenerate attention and verifying enc_attn loss is working.
    """

    def __init__(self, log_every_n_epochs=5, num_samples=2):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self._batch = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if batch_idx == 0 and self._batch is None:
            n = self.num_samples
            self._batch = {
                'laser_spec':   batch['laser_spec'][:n].cpu(),
                'padding_mask': batch['padding_mask'][:n].cpu()
                                if batch.get('padding_mask') is not None else None,
                'audio_path':   batch['audio_path'][:n]
                                if batch.get('audio_path') is not None else None,
            }

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        if self._batch is None:
            return
        if not getattr(pl_module, 'w_enc_attn', 0.0) > 0:
            return
        if trainer.logger is None:
            return

        device = pl_module.device
        student = pl_module.ema.ema_model if pl_module.use_ema else pl_module.system.student
        teacher_encoder = pl_module.system.teacher.encoder
        blocks = pl_module.enc_attn_blocks

        laser = self._batch['laser_spec'].to(device)
        mask  = self._batch['padding_mask'].to(device) \
                if self._batch['padding_mask'] is not None else None
        audio_paths = self._batch['audio_path']

        try:
            pl_module.eval()
            with torch.no_grad():
                # Student forward → populates _last_attn_weights on student encoder blocks
                student(laser, padding_mask=mask)

                # Teacher forward on audio → populates _last_attn_weights on teacher encoder
                teacher_maps = {}
                if audio_paths is not None:
                    path = audio_paths[0]
                    audio = whisper.load_audio(path)
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio, n_mels=80).unsqueeze(0).to(device)
                    teacher_encoder(mel)
                    for b_idx in blocks:
                        attn = teacher_encoder.blocks[b_idx].attn
                        if hasattr(attn, '_last_attn_weights'):
                            teacher_maps[b_idx] = attn._last_attn_weights[0].float().cpu().numpy()

            log_dict = {}
            for b_idx in blocks:
                s_attn = student.encoder.blocks[b_idx].attn
                if not hasattr(s_attn, '_last_attn_weights'):
                    continue
                s_map = s_attn._last_attn_weights[0].float().cpu().numpy()  # [T, T]

                has_teacher = b_idx in teacher_maps
                fig, axes = plt.subplots(1, 2 if has_teacher else 1,
                                         figsize=(14 if has_teacher else 7, 5))
                if not has_teacher:
                    axes = [axes]

                axes[0].imshow(s_map, aspect='auto', origin='upper', cmap='hot',
                               vmin=0, vmax=s_map.max())
                axes[0].set_title(f'Student block {b_idx} — ep{epoch}')
                axes[0].set_xlabel('Key position')
                axes[0].set_ylabel('Query position')

                if has_teacher:
                    t_map = teacher_maps[b_idx]
                    axes[1].imshow(t_map, aspect='auto', origin='upper', cmap='hot',
                                   vmin=0, vmax=t_map.max())
                    axes[1].set_title(f'Teacher block {b_idx}')
                    axes[1].set_xlabel('Key position')

                plt.tight_layout()
                log_dict[f'attn_maps/block_{b_idx}'] = wandb.Image(fig)
                plt.close(fig)

            if log_dict:
                trainer.logger.experiment.log(log_dict, step=trainer.global_step)
                print(f"[AttnMaps] Logged {len(log_dict)} attention map(s) at epoch {epoch}")

        except Exception as e:
            import traceback
            print(f"[LogAttnMapsCallback] ERROR: {e}")
            traceback.print_exc()
        finally:
            pl_module.train()
