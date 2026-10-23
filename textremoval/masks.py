"""Text masks in pixel space and on the latent grid."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TEXT_SCOPES = ("largest_text", "all_text")


def load_binary_mask(path, threshold: int = 128) -> np.ndarray:
    with Image.open(path) as opened:
        gray = np.asarray(opened.convert("L"), dtype=np.uint8)
    return gray >= threshold


def select_text_mask(mask: np.ndarray, scope: str) -> np.ndarray:
    """``all_text``: the whole mask. ``largest_text``: its largest 8-connected component (first in scan order on
    ties)."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Text mask must be two-dimensional; got {binary.shape}.")
    if not binary.any():
        raise ValueError("Text mask is empty.")
    if scope == "all_text":
        return binary.copy()
    if scope != "largest_text":
        raise ValueError(f"Unknown text scope: {scope}")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError("Text mask has no foreground connected component.")
    # Labels are assigned in scan order; np.argmax keeps the first label on ties.
    label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == label


def tight_box(mask: np.ndarray) -> list[int]:
    """XYXY box (exclusive max) of the ``True`` pixels."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def resize_mask_to_geometry(mask: np.ndarray, geometry: Any) -> np.ndarray:
    """Apply the resize/crop RevealLayer uses for images (``runtime.ImageGeometry``) to a mask, nearest-neighbour."""
    mask = np.asarray(mask, dtype=bool)
    orig_w, orig_h = geometry.original_size
    if mask.shape != (orig_h, orig_w):
        raise ValueError(f"mask shape {mask.shape} does not match original_size {(orig_h, orig_w)}.")
    content_w, content_h = geometry.content_size
    target_w, target_h = geometry.target_size
    offset_x, offset_y = geometry.offset
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = pil_mask.resize((content_w, content_h), Image.Resampling.NEAREST)
    cropped = resized.crop((offset_x, offset_y, offset_x + target_w, offset_y + target_h))
    return np.asarray(cropped, dtype=np.uint8) >= 128


def pixel_mask_to_latent(mask: np.ndarray, vae_scale_factor: int = 8) -> torch.Tensor:
    """Latent cells that contain any mask pixel, as a bool ``[1, 1, h, w]`` tensor."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    if height % vae_scale_factor or width % vae_scale_factor:
        raise ValueError(f"mask shape {mask.shape} must be divisible by {vae_scale_factor}.")
    pooled = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], vae_scale_factor)
    return pooled > 0.5


def box_mask(box_xyxy, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
    """Bool ``[1, 1, h, w]`` mask of the latent cells inside a pixel-space BBox."""
    x1, y1, x2, y2 = (int(value) // vae_scale_factor for value in box_xyxy)
    mask = torch.zeros((1, 1, height, width), dtype=torch.bool)
    mask[..., y1:y2, x1:x2] = True
    return mask


def dilate_mask(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Square dilation of a bool ``[B, 1, h, w]`` mask by ``cells`` latent cells."""
    if cells < 0:
        raise ValueError("cells must be non-negative.")
    if cells == 0:
        return mask
    return F.max_pool2d(mask.float(), 2 * cells + 1, stride=1, padding=cells) > 0.5
