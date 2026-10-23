"""Single-pass text removal: one RevealLayer inference, text layer from image A, background kept to image B.

Inputs of one inference (all at the RevealLayer resolution):

* image B: the original image. Its latent ``z_src`` is the keep target of the background outside the text.
* full-image layer (index 0, also the background layer's refiner condition): image B with the text mask (dilated
  by ``inpaint_dilation_px``) filled from the surrounding colours (``fill_from_boundary``).
* image A: the original inside the text mask, the surrounding colours outside it (``extend_boundary_colors``).
  It is the refiner condition of the text layer and a second full-image layer that only the text layer attends to,
  in place of layer 0, so the text layer is extracted from an image that contains only the text.
* the text BBox as the single object layer (index 2).

During denoising the ``TextRemovalController`` blocks the full-image tokens under the current text mask from the
background queries and keeps the background equal to ``z_src`` outside it (see ``textremoval/guidance.py``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from textremoval.fill import extend_boundary_colors, fill_from_boundary
from textremoval.guidance import BACKGROUND_LAYER, TEXT_LAYER, TextRemovalController
from textremoval.masks import box_mask, pixel_mask_to_latent, resize_mask_to_geometry, tight_box
from textremoval.runtime import (
    PROMPT,
    VAE_SCALE_FACTOR,
    encode_latent,
    grey_latent,
    image_to_tensor,
    latent_to_image,
    prepare_image_and_boxes,
    seed_everything,
)


@dataclass
class TextRemovalConfig:
    seed: int = 43
    steps: int = 30
    cfg: float = 1.0
    max_layer: int = 12
    keep_tau: float = 0.0              # keep threshold (0 = replace every differing cell outside the text)
    fg_tau: float = 0.9                # grey-distance threshold of the text mask M_k
    text_mask_steps: int = 1           # steps (from 0) that use the input text mask as M_k
    keep_dilation: int = 1             # latent cells between M_k and the keep region
    inpaint_dilation_px: int = 2       # dilation of the text mask before the full-image layer is filled
    gradual_blur_max_sigma: float = 12.0
    gradual_blur_distance_px: float = 60.0
    preview_every: int = 3             # with save_intermediates: clean-estimate previews every N steps


@dataclass
class PreparedInputs:
    image_a: Image.Image
    image_b: Image.Image
    full_image: Image.Image
    boxes: list
    geometry: object
    text_mask: np.ndarray              # at the RevealLayer resolution
    native_bbox: list


def prepare_inputs(image: Image.Image, text_mask: np.ndarray, config: TextRemovalConfig) -> PreparedInputs:
    """``image`` and ``text_mask`` (bool, same size) at native resolution -> the inputs of one inference."""
    image = image.convert("RGB")
    if text_mask.shape != (image.height, image.width):
        raise ValueError(f"text mask {text_mask.shape} does not match image {(image.height, image.width)}.")
    bbox = tight_box(text_mask)
    image_a_native = extend_boundary_colors(image, text_mask, config.gradual_blur_max_sigma,
                                            gradual_distance_px=config.gradual_blur_distance_px)
    image_b, boxes, geometry = prepare_image_and_boxes(image, [bbox], max_layers=config.max_layer)
    image_a, boxes_a, geometry_a = prepare_image_and_boxes(image_a_native, [bbox], max_layers=config.max_layer)
    if boxes_a != boxes or geometry_a != geometry or len(boxes) != TEXT_LAYER + 1:
        raise RuntimeError(f"expected one aligned text BBox, got {boxes}.")
    target_mask = resize_mask_to_geometry(text_mask, geometry)
    full_image = fill_from_boundary(image_b, target_mask, dilation_px=config.inpaint_dilation_px,
                                    sigma=config.gradual_blur_max_sigma,
                                    gradual_distance_px=config.gradual_blur_distance_px)
    return PreparedInputs(image_a, image_b, full_image, boxes, geometry, target_mask, bbox)


@torch.no_grad()
def remove_text(pipeline, decoder, device, inputs: PreparedInputs, config: TextRemovalConfig, *,
                record_previews: bool = False):
    """Run one inference. Returns ``(background RGB, background RGBA, text layer RGBA, controller, seconds)``."""
    width, height = inputs.geometry.target_size
    z_grey = grey_latent(pipeline, height, width, device)
    source = encode_latent(pipeline, inputs.image_b, device)
    latent_h, latent_w = height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR
    previews = set(range(0, config.steps, config.preview_every)) | {config.steps - 1} if record_previews else ()
    controller = TextRemovalController(
        text_latent=pixel_mask_to_latent(inputs.text_mask, VAE_SCALE_FACTOR).to(device),
        box=box_mask(inputs.boxes[TEXT_LAYER], latent_h, latent_w).to(device), z_grey=z_grey, source=source,
        num_layers=len(inputs.boxes), keep_tau=config.keep_tau, fg_tau=config.fg_tau,
        text_mask_steps=config.text_mask_steps, keep_dilation=config.keep_dilation, preview_steps=previews)
    image_a = image_to_tensor(inputs.image_a, device)

    seed_everything(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    images, rgba, _, _ = pipeline(
        prompt=PROMPT, full_image=image_to_tensor(inputs.full_image, device), validation_box=list(inputs.boxes),
        generator=generator, height=height, width=width, num_layers=len(inputs.boxes),
        guidance_scale=config.cfg, num_inference_steps=config.steps, transparent_decoder=decoder,
        return_dict=False,
        blocked_token_maps_callback=controller.token_maps,
        velocity_correction_callback=controller.correct,
        foreground_adapter_image=image_a,
        foreground_reference_image=image_a,
    )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    background_rgba = Image.fromarray(rgba[BACKGROUND_LAYER], "RGBA")
    text_rgba = Image.fromarray(rgba[TEXT_LAYER], "RGBA")
    return images[BACKGROUND_LAYER], background_rgba, text_rgba, controller, seconds


def save_result(out_dir: Path, pipeline, inputs: PreparedInputs, result, config: TextRemovalConfig, *,
                save_intermediates: bool = False, extra_metadata: dict | None = None) -> None:
    background, background_rgba, text_rgba, controller, seconds = result
    out_dir.mkdir(parents=True, exist_ok=True)
    background.save(out_dir / "bg_rgb.png")
    background_rgba.save(out_dir / "bg_rgba.png")
    text_rgba.save(out_dir / "text_layer_rgba.png")
    if save_intermediates:
        inputs.image_a.save(out_dir / "image_a.png")
        inputs.full_image.save(out_dir / "full_image.png")
        np.save(out_dir / "keep_masks.npy", torch.stack(controller.keep_masks).cpu().numpy())
        np.save(out_dir / "block_masks.npy", torch.stack(controller.block_masks).cpu().numpy())
        with (out_dir / "step_log.csv").open("w", encoding="utf-8") as handle:
            keys = list(controller.log[0])
            handle.write(",".join(keys) + "\n")
            handle.writelines(",".join(str(row[key]) for key in keys) + "\n" for row in controller.log)
        previews = out_dir / "previews"
        previews.mkdir(exist_ok=True)
        for layer, latents in controller.previews.items():
            for step, latent in sorted(latents.items()):
                latent_to_image(pipeline, latent).save(previews / f"{layer}_clean_step_{step:02d}.png")
    metadata = {
        **(extra_metadata or {}),
        "seconds": round(seconds, 2),
        "boxes": [list(box) for box in inputs.boxes],
        "native_bbox": inputs.native_bbox,
        "original_size": list(inputs.geometry.original_size),
        "target_size": list(inputs.geometry.target_size),
        "config": asdict(config),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
