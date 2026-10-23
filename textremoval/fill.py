"""Pixel-space fills from the colours around a mask."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def extend_boundary_colors(
    image: Image.Image, mask: np.ndarray, sigma: float, *, gradual_distance_px: float | None = None,
    blur_levels: int = 9,
) -> Image.Image:
    """Keep ``image`` inside ``mask``; outside, spread the mask's outer-edge colours and blur them.

    The outer edge is the ring of non-mask pixels 8-adjacent to the mask. Every
    pixel takes the colour of its nearest ring pixel (Euclidean distance); the
    resulting field is blurred with a Gaussian of ``sigma`` pixels (0 = no blur).
    Because the field is defined everywhere from ring colours only, the blur
    never mixes the mask's own colours into the outside.

    With ``gradual_distance_px`` the blur grows with the distance ``d`` to the
    mask: sigma(d) = ``sigma`` * min(d / gradual_distance_px, 1), realised by
    linear interpolation between ``blur_levels`` uniformly spaced blur levels.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != array.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match image {array.shape[:2]}.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not mask.any():
        raise ValueError("mask is empty; there is no boundary colour to extend.")
    ring = ~mask & (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if not ring.any():  # the mask covers the whole image
        return Image.fromarray(array.copy(), mode="RGB")
    # Labels index the zero pixels of the source, i.e. the ring pixels.
    _, labels = cv2.distanceTransformWithLabels(
        (~ring).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ring_y, ring_x = np.nonzero(ring)
    palette = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    palette[labels[ring_y, ring_x]] = array[ring_y, ring_x]
    field = palette[labels]
    if sigma > 0 and gradual_distance_px is None:
        field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), sigma)
        field = np.clip(np.rint(field), 0, 255).astype(np.uint8)
    elif sigma > 0:
        if gradual_distance_px <= 0 or blur_levels < 2:
            raise ValueError("gradual_distance_px must be positive and blur_levels at least 2.")
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        position = np.clip(distance / gradual_distance_px, 0.0, 1.0) * (blur_levels - 1)
        field_f = field.astype(np.float32)
        stack = np.stack([field_f] + [
            cv2.GaussianBlur(field_f, (0, 0), level_sigma)
            for level_sigma in np.linspace(0.0, sigma, blur_levels)[1:]
        ])  # [levels, H, W, 3]
        lower = np.minimum(np.floor(position).astype(np.int64), blur_levels - 2)
        weight = (position - lower)[..., None]
        rows, cols = np.indices(lower.shape)
        blended = stack[lower, rows, cols] * (1.0 - weight) + stack[lower + 1, rows, cols] * weight
        field = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    result = np.where(mask[..., None], array, field)
    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def fill_from_boundary(image: Image.Image, mask: np.ndarray, *, dilation_px: int, sigma: float,
                       gradual_distance_px: float) -> Image.Image:
    """Fill ``mask`` (dilated by ``dilation_px``) with the colours of the pixels just outside it, blurred more the
    further inside; outside the dilated mask ``image`` is unchanged."""
    region = np.asarray(mask, dtype=bool)
    if dilation_px:
        kernel = np.ones((2 * dilation_px + 1,) * 2, np.uint8)
        region = cv2.dilate(region.astype(np.uint8), kernel) > 0
    return extend_boundary_colors(image, ~region, sigma, gradual_distance_px=gradual_distance_px)
