import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarKDSystem(nn.Module):
    def __init__(self, student_model, teacher_model, teacher_type="whisper",
                 distillation_mode="single", skip_decoder=False):
        """
        Args:
            student_model: the radar/LDV student encoder
            teacher_model: frozen Whisper model (can be None for pretrain)
            teacher_type: "whisper"
            distillation_mode: "single", "multi_layer", or "component_wise"
                - single: Only final encoder output (default, backward compatible)
                - multi_layer: Intermediate block outputs + final
                - component_wise: Stem + Attention + Blocks + Final
            skip_decoder: If True, skip decoder forward pass (encoder-only mode)
        """
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model  # Accept pre-loaded teacher model (can be None for pretrain)
        self.teacher_type = teacher_type  # always "whisper"
        self.distillation_mode = distillation_mode
        self.skip_decoder = skip_decoder

        # Freeze entire teacher model (already loaded and on correct device)
        # Only if teacher is provided (not None in pretrain modes)
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()  # Keep in eval mode
            print(f"KD System initialized with {teacher_type.upper()} {'encoder' if skip_decoder else 'decoder'}")
            print(f"Distillation mode: {distillation_mode.upper()}")
            if skip_decoder:
                print("✓ Decoder SKIPPED (encoder-only mode)")
            
    def forward(self, radar_input, teacher_feats, decoder_input_ids, padding_mask, return_ctc=False):
        """
        Args:
            teacher_feats: Can be:
                - Tensor [B, 1500, teacher_dim] (single-layer, backward compatible)
                - Dict with keys: 'final', 'blocks' (for multi_layer/component_wise)
            return_ctc: If True, also return CTC logits from student encoder

        Returns:
            Dict with keys:
                - student_feats: Final projected student features [B, T, teacher_dim]
                - teacher_feats: Final teacher features [B, T, teacher_dim]
                - student_logits: Student decoder logits (if teacher is not None)
                - teacher_logits: Teacher decoder logits (if teacher is not None)
                - student_feats_intermediate: Dict with 'blocks_projected' for multi-layer
                - teacher_feats_intermediate: List of teacher block outputs for multi-layer
                - ctc_logits: CTC logits (if return_ctc=True)
        """
        # --- B. Student Branch (Always run) ---
        # Extract features based on distillation mode
        return_intermediate = (self.distillation_mode in ["multi_layer", "component_wise"])

        if return_ctc:
            student_result = self.student(radar_input, padding_mask,
                                         return_ctc=True,
                                         return_intermediate=return_intermediate)
            if return_intermediate:
                student_feats_final, student_feats_intermediate, ctc_logits = student_result
            else:
                student_feats_final, ctc_logits = student_result
                student_feats_intermediate = None
        else:
            student_result = self.student(radar_input, padding_mask,
                                         return_ctc=False,
                                         return_intermediate=return_intermediate)
            if return_intermediate:
                student_feats_final, student_feats_intermediate = student_result
            else:
                student_feats_final = student_result
                student_feats_intermediate = None
            ctc_logits = None

        # --- CTC-only mode: Skip teacher branch entirely ---
        if self.teacher is None:
            output = {
                "student_feats": student_feats_final,
                "teacher_feats": None,  # No teacher in CTC-only mode
                "student_logits": None,  # No decoder in CTC-only mode
                "teacher_logits": None
            }
            # Add intermediate features if using multi-layer/component-wise
            if student_feats_intermediate is not None:
                output["student_feats_intermediate"] = student_feats_intermediate
            if return_ctc:
                output["ctc_logits"] = ctc_logits
            return output

        # --- Encoder-only mode: Skip decoder entirely ---
        if self.skip_decoder:
            # Get teacher features and move to correct device
            teacher_device = next(self.teacher.parameters()).device
            
            # Handle backward compatibility: teacher_feats can be Tensor or Dict
            if isinstance(teacher_feats, dict):
                teacher_feats_final = teacher_feats['final'].to(teacher_device)
                teacher_feats_intermediate = teacher_feats.get('blocks', None)
                if teacher_feats_intermediate is not None:
                    teacher_feats_intermediate = [f.to(teacher_device) for f in teacher_feats_intermediate]
            else:
                teacher_feats_final = teacher_feats.to(teacher_device)
                teacher_feats_intermediate = None
            
            student_feats_final = student_feats_final.to(teacher_device)
            
            output = {
                "student_feats": student_feats_final,
                "teacher_feats": teacher_feats_final,
                "student_logits": None,  # No decoder
                "teacher_logits": None   # No decoder
            }
            
            # Add intermediate features if using multi-layer/component-wise
            if student_feats_intermediate is not None:
                output["student_feats_intermediate"] = student_feats_intermediate
            if teacher_feats_intermediate is not None:
                output["teacher_feats_intermediate"] = teacher_feats_intermediate
            if return_ctc:
                output["ctc_logits"] = ctc_logits
            
            return output

        # --- Full KD mode: Run teacher branch ---
        # Get teacher decoder's device and ensure all inputs match
        teacher_device = next(self.teacher.parameters()).device

        # Handle backward compatibility: teacher_feats can be Tensor or Dict
        if isinstance(teacher_feats, dict):
            # New format: Dict with 'final', 'blocks', etc.
            teacher_feats_final = teacher_feats['final'].to(teacher_device)
            teacher_feats_intermediate = teacher_feats.get('blocks', None)
            if teacher_feats_intermediate is not None:
                teacher_feats_intermediate = [f.to(teacher_device) for f in teacher_feats_intermediate]
        else:
            # Backward compatible: Tensor (single final layer)
            teacher_feats_final = teacher_feats.to(teacher_device)
            teacher_feats_intermediate = None

        decoder_input_ids = decoder_input_ids.to(teacher_device)

        # --- Prepare padding mask for decoder (CRITICAL for fair comparison) ---
        # Both teacher and student decoders must see the SAME masked inputs
        # Otherwise KD loss compares incompatible distributions!
        padding_mask_decoder = None
        if padding_mask is not None:
            padding_mask_decoder = padding_mask.to(teacher_device)
            # Downsample mask from 3000 -> 1500 if needed (match encoder output)
            if padding_mask_decoder.shape[1] == 3000:
                padding_mask_decoder = padding_mask_decoder[:, ::2]
            # Ensure mask matches sequence length (should be 1500)
            if padding_mask_decoder.shape[1] != teacher_feats_final.shape[1]:
                padding_mask_decoder = padding_mask_decoder[:, :teacher_feats_final.shape[1]]

        # --- A. Teacher Branch ---
        with torch.no_grad():
            # WE SKIP THE ENCODER!
            # teacher_feats_final is already precomputed and passed in.

            # CRITICAL: Zero out padded frames for teacher (for fair comparison)
            teacher_feats_for_decoder = teacher_feats_final
            if padding_mask_decoder is not None:
                teacher_feats_for_decoder = teacher_feats_final * padding_mask_decoder.unsqueeze(-1)

            # Run decoder
            teacher_logits = self.teacher.decoder(decoder_input_ids, teacher_feats_for_decoder)

        student_feats_final = student_feats_final.to(teacher_device)

        # Normalize first (on valid + padding tokens alike — per-token LN is safe),
        # then zero out padding. This avoids normalizing already-zeroed vectors which
        # produces 0/eps artifacts. IMPORTANT: keep original student_feats_final for feature loss.
        student_feats_for_decoder = F.layer_norm(
            student_feats_final.float(), [student_feats_final.shape[-1]]
        ).to(student_feats_final.dtype)

        if padding_mask_decoder is not None:
            student_feats_for_decoder = student_feats_for_decoder * padding_mask_decoder.unsqueeze(-1)

        # Run student through decoder
        student_logits = self.teacher.decoder(decoder_input_ids, student_feats_for_decoder)

        output = {
            "student_feats": student_feats_final,  # Original (non-zeroed) for feature loss
            "teacher_feats": teacher_feats_final,  # Pass through for loss
            "student_logits": student_logits,
            "teacher_logits": teacher_logits
        }

        # Add intermediate features if using multi-layer/component-wise distillation
        if student_feats_intermediate is not None:
            output["student_feats_intermediate"] = student_feats_intermediate
        if teacher_feats_intermediate is not None:
            output["teacher_feats_intermediate"] = teacher_feats_intermediate

        if return_ctc:
            output["ctc_logits"] = ctc_logits

        return output