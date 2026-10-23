"""Per-step guidance of the RevealLayer denoising loop for text removal.

At step ``k`` the text mask ``M_k`` (latent cells) is the input text mask at ``k < text_mask_steps`` and afterwards
the cells where the text layer's clean estimate differs from the transparent (grey) latent by more than ``fg_tau``,
inside the text BBox. The controller is handed to the pipeline as two callbacks:

* ``correct`` (``velocity_correction_callback``, after the transformer at step ``k``): outside ``M_k`` dilated by
  ``keep_dilation`` cells, the background layer's clean estimate is replaced by the latent of the original image
  ``z_src`` wherever it strayed more than ``keep_tau`` (``keep_tau = 0``: everywhere it differs). Nothing is done
  inside the dilated ``M_k``, where the background is generated.
* ``token_maps`` (``blocked_token_maps_callback``, before the transformer at step ``k + 1``): the full-image layer's
  tokens under ``M_k`` are hidden from the background queries, so the background never reads the text pixels.
  The blocking therefore uses the mask of the previous step; step 0 uses the input text mask.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from textremoval.masks import dilate_mask

FULL_IMAGE_LAYER = 0
BACKGROUND_LAYER = 1
TEXT_LAYER = 2
_SOURCE_RMS_FLOOR = 1e-6


def grey_distance(clean: torch.Tensor, grey_latent: torch.Tensor) -> torch.Tensor:
    """``sqrt(mean_c (clean - grey)^2)`` for ``[B, C, h, w]`` inputs, as ``[B, 1, h, w]`` float32."""
    return (clean.float() - grey_latent.float()).square().mean(dim=1, keepdim=True).sqrt()


def source_rms(source_latent: torch.Tensor, keep_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image RMS of ``z_src`` over the keep region (``[B]``, floored at 1e-6) and whether the region is
    non-empty."""
    source = source_latent.float()
    mask = keep_mask.float().to(source.device).expand(source.shape[0], 1, *source.shape[-2:])
    keep_count = mask.sum(dim=(1, 2, 3))
    sum_sq = (mask * source.square()).sum(dim=(1, 2, 3))
    rms = torch.sqrt(sum_sq / (source.shape[1] * keep_count).clamp_min(1.0))
    return rms.clamp_min(_SOURCE_RMS_FLOOR), keep_count > 0


def replace_with_source(
    velocity: torch.Tensor,
    latents: torch.Tensor,
    sigma: Any,
    source_latent: torch.Tensor,
    keep_mask: torch.Tensor,
    tau: float,
    *,
    layer_index: int = BACKGROUND_LAYER,
) -> tuple[torch.Tensor, float]:
    """Replace one layer's clean estimate by ``z_src`` inside ``keep_mask`` where it strayed too far.

    In float32::

        z0_hat = z - sigma * v,   d = sqrt(mean_c (z0_hat - z_src)^2) / rms(z_src over the keep region)
        R      = keep_mask & (d > tau)
        v'     = v + R * (z0_hat - z_src) / sigma

    so that ``z - sigma * v' == z_src`` on ``R``. Every other position and layer is returned unchanged.
    Returns the velocity and the selected fraction of the keep region.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    batch, _, _, height, width = velocity.shape
    rms, has_keep = source_rms(source_latent, keep_mask)
    device = latents.device
    sigma_value = float(sigma.detach().float()) if isinstance(sigma, torch.Tensor) else float(sigma)
    z_layer = latents[:, layer_index].float()
    v_layer = velocity[:, layer_index].float()
    mask = keep_mask.to(device).expand(batch, 1, height, width) > 0.5
    mask = mask & has_keep.to(device).view(batch, 1, 1, 1)

    diff = (z_layer - sigma_value * v_layer) - source_latent.float().to(device)
    distance = torch.sqrt(diff.square().mean(dim=1, keepdim=True)) / rms.to(device).view(batch, 1, 1, 1)
    selected = mask & (distance > tau)
    keep_count, selected_count = int(mask.sum()), int(selected.sum())
    fraction = selected_count / keep_count if keep_count else 0.0
    if sigma_value <= 0 or selected_count == 0:
        return velocity, fraction

    selected = selected.expand_as(v_layer)
    v_guided = v_layer + selected.float() * diff / sigma_value
    corrected = velocity.clone()
    corrected[:, layer_index] = torch.where(selected, v_guided.to(velocity.dtype), velocity[:, layer_index])
    return corrected, fraction


class TextRemovalController:
    """Holds ``M_k`` between the velocity callback (step ``k``) and the token-map callback (step ``k + 1``)."""

    def __init__(self, *, text_latent: torch.Tensor, box: torch.Tensor, z_grey: torch.Tensor,
                 source: torch.Tensor, num_layers: int, keep_tau: float = 0.0, fg_tau: float = 0.9,
                 text_mask_steps: int = 1, keep_dilation: int = 1, preview_steps=()):
        self.text, self.box, self.z_grey, self.source = text_latent, box, z_grey, source
        self.num_layers, self.keep_tau, self.fg_tau = num_layers, keep_tau, fg_tau
        self.text_mask_steps, self.keep_dilation = text_mask_steps, keep_dilation
        self.block_mask = text_latent                  # used by step 0
        self.keep_masks: list[torch.Tensor] = []
        self.block_masks: list[torch.Tensor] = []
        self.log: list[dict[str, Any]] = []
        self.preview_steps = set(preview_steps)
        self.previews: dict[str, dict[int, torch.Tensor]] = {"bg": {}, "fg": {}}

    def token_maps(self, step: int) -> torch.Tensor:
        """``blocked_token_maps_callback``: the full-image layer's tokens under the latest ``M``."""
        self.block_masks.append(self.block_mask[0, 0].clone())
        tokens = F.max_pool2d(self.block_mask.float(), 2) > 0.5          # [1, 1, h/2, w/2]
        maps = torch.zeros((1, self.num_layers, *tokens.shape[-2:]), dtype=torch.bool, device=tokens.device)
        maps[0, FULL_IMAGE_LAYER] = tokens[0, 0]
        return maps

    def correct(self, step: int, sigma: torch.Tensor, latents: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """``velocity_correction_callback``: ``M_k`` from this step's text-layer estimate, keep outside it."""
        sigma_value = float(sigma)
        fg_clean = latents[:, TEXT_LAYER].float() - sigma_value * velocity[:, TEXT_LAYER].float()
        if step < self.text_mask_steps:
            mask = self.text
        else:
            mask = (grey_distance(fg_clean, self.z_grey) > self.fg_tau) & self.box
        keep = ~dilate_mask(mask, self.keep_dilation)
        velocity, replaced = replace_with_source(velocity, latents, sigma, self.source, keep.float(), self.keep_tau)
        self.log.append({"step": step, "sigma": sigma_value, "text_mask_cells": int(mask.sum()),
                         "block_mask_cells": int(self.block_masks[-1].sum()) if self.block_masks else None,
                         "keep_cells": int(keep.sum()), "keep_replaced_fraction": replaced})
        self.keep_masks.append(mask[0, 0].clone())
        self.block_mask = mask                                            # blocks step k + 1
        if step in self.preview_steps:
            # The text layer only exists inside its BBox; show grey (transparent) outside it.
            self.previews["fg"][step] = torch.where(self.box, fg_clean, self.z_grey).to(latents.dtype)
            self.previews["bg"][step] = (latents[:, BACKGROUND_LAYER].float()
                                         - sigma_value * velocity[:, BACKGROUND_LAYER].float()).to(latents.dtype)
        return velocity
