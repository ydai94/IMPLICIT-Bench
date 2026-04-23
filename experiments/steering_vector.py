"""
Steering vector computation and steered image generation for QwenImagePipeline.

Provides three steering modes:
  1. Full-prompt / full-triple SV:  d = encode(anti) - encode(stereo)
  2. Tail-only SV:  mean-pool tail embeddings, broadcast across positions
  3. KG-triple SV:  same as (1) but with explicit CPU/device handling

Adapted from stereoset/steering_experiment/steering_vector.py,
run_tail_experiment.py, and run_kg_steering.py.
"""

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from config import IMAGE_MODEL_PATH, NUM_INFERENCE_STEPS, CFG_SCALE


class SteeringVectorEditor:
    def __init__(self, model_path=IMAGE_MODEL_PATH, device="cuda",
                 use_cpu_offload=False):
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if use_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, prompt):
        """Encode a single prompt, return (embeds, mask)."""
        embeds, mask = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.pipe.device,
        )
        if mask is None:
            mask = torch.ones(embeds.shape[:2], device=embeds.device,
                              dtype=torch.long)
        return embeds, mask

    # ------------------------------------------------------------------
    # Steering vector computation
    # ------------------------------------------------------------------

    def compute_steering_vector(self, stereo_prompt, anti_stereo_prompt):
        """Full-prompt steering vector: d = encode(anti) - encode(stereo).

        Handles sequence length mismatch by padding shorter to match longer.
        Returns (steering_vector, steering_mask) with shape (1, max_seq, 3584).
        """
        with torch.no_grad():
            e_stereo, m_stereo = self._encode(stereo_prompt)
            e_anti, m_anti = self._encode(anti_stereo_prompt)

        # Move to CPU float for consistent computation
        e_stereo, m_stereo = e_stereo.cpu().float(), m_stereo.cpu().float()
        e_anti, m_anti = e_anti.cpu().float(), m_anti.cpu().float()

        max_seq = max(e_stereo.shape[1], e_anti.shape[1])

        if e_stereo.shape[1] < max_seq:
            pad = max_seq - e_stereo.shape[1]
            e_stereo = F.pad(e_stereo, (0, 0, 0, pad))
            m_stereo = F.pad(m_stereo, (0, pad), value=0)
        if e_anti.shape[1] < max_seq:
            pad = max_seq - e_anti.shape[1]
            e_anti = F.pad(e_anti, (0, 0, 0, pad))
            m_anti = F.pad(m_anti, (0, pad), value=0)

        d = e_anti - e_stereo
        combined_mask = (m_stereo + m_anti).clamp(max=1)
        return d, combined_mask

    def compute_tail_steering_vector(self, stereo_tail, anti_stereo_tail):
        """Tail-only steering vector: mean-pool tail embeddings to (1, 1, 3584)."""
        with torch.no_grad():
            e_stereo, m_stereo = self._encode(stereo_tail)
            e_anti, m_anti = self._encode(anti_stereo_tail)

        e_stereo, m_stereo = e_stereo.cpu().float(), m_stereo.cpu().float()
        e_anti, m_anti = e_anti.cpu().float(), m_anti.cpu().float()

        stereo_pooled = (
            (e_stereo * m_stereo.unsqueeze(-1)).sum(dim=1, keepdim=True)
            / m_stereo.sum(dim=1, keepdim=True).unsqueeze(-1)
        )
        anti_pooled = (
            (e_anti * m_anti.unsqueeze(-1)).sum(dim=1, keepdim=True)
            / m_anti.sum(dim=1, keepdim=True).unsqueeze(-1)
        )

        d = anti_pooled - stereo_pooled  # (1, 1, 3584)
        return d

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _align_steering_to_target(self, steering_vec, steering_mask, target_seq_len):
        """Pad or truncate steering vector to match target sequence length."""
        sv_len = steering_vec.shape[1]
        if sv_len == target_seq_len:
            return steering_vec, steering_mask
        elif sv_len < target_seq_len:
            pad = target_seq_len - sv_len
            sv = F.pad(steering_vec, (0, 0, 0, pad))
            sm = F.pad(steering_mask, (0, pad), value=0)
            return sv, sm
        else:
            return (steering_vec[:, :target_seq_len, :],
                    steering_mask[:, :target_seq_len])

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_with_steering(self, neutral_prompt, steering_vector, steering_mask,
                               alpha, seed, num_steps=NUM_INFERENCE_STEPS,
                               cfg_scale=CFG_SCALE):
        """Generate image from neutral prompt with full steering vector applied."""
        with torch.no_grad():
            prompt_embeds, prompt_mask = self._encode(neutral_prompt)

        target_len = prompt_embeds.shape[1]
        sv_aligned, _ = self._align_steering_to_target(
            steering_vector, steering_mask, target_len
        )

        sv_aligned = sv_aligned.to(device=prompt_embeds.device,
                                   dtype=prompt_embeds.dtype)
        steered_embeds = prompt_embeds + alpha * sv_aligned

        # Ensure CUDA for the transformer
        steered_embeds = steered_embeds.to(device="cuda", dtype=torch.bfloat16)
        prompt_mask = prompt_mask.to(device="cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt_embeds=steered_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt=" ",
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_with_tail_steering(self, neutral_prompt, tail_steering_dir,
                                    alpha, seed, num_steps=NUM_INFERENCE_STEPS,
                                    cfg_scale=CFG_SCALE):
        """Generate image with tail-only steering (broadcast across all positions)."""
        with torch.no_grad():
            prompt_embeds, prompt_mask = self._encode(neutral_prompt)

        target_len = prompt_embeds.shape[1]
        sv_broadcast = tail_steering_dir.to(device=prompt_embeds.device,
                                            dtype=prompt_embeds.dtype)
        sv_broadcast = sv_broadcast.expand(-1, target_len, -1)

        steered_embeds = prompt_embeds + alpha * sv_broadcast

        steered_embeds = steered_embeds.to(device="cuda", dtype=torch.bfloat16)
        prompt_mask = prompt_mask.to(device="cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt_embeds=steered_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt=" ",
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_baseline(self, prompt, seed, num_steps=NUM_INFERENCE_STEPS,
                          cfg_scale=CFG_SCALE):
        """Generate image from a text prompt without steering."""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=generator,
        )
        return result.images[0]
