"""Model loading, image geometry and latent helpers shared by the text-removal commands."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from textremoval import REPO_ROOT

PROMPT = "Decompose the image into foreground and background."
VAE_SCALE_FACTOR = 8


@dataclass(frozen=True)
class ModelPaths:
    flux: Path
    reveallayer: Path
    transparent_decoder: Path
    vit32: Path


@dataclass(frozen=True)
class ImageGeometry:
    original_size: tuple[int, int]
    content_size: tuple[int, int]
    target_size: tuple[int, int]
    offset: tuple[int, int]


def model_paths(models_dir: Path | str) -> ModelPaths:
    root = Path(models_dir).expanduser().resolve()
    return ModelPaths(
        flux=root / "FLUX.1-dev",
        reveallayer=root / "RevealLayer",
        transparent_decoder=root / "RevealLayer/xvae/transparent_decoder_ckpt.pth",
        vit32=root / "ViT-B-32/vit_b_32-d86f8d99.pth",
    )


def validate_model_paths(paths: ModelPaths) -> None:
    missing = [path for path in paths.__dict__.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required model files/directories are missing (see README):\n{formatted}")


def load_models(models_dir: Path | str, *, gpu_id: int = 0, max_layer: int = 12,
                cfg_path: Path | str = REPO_ROOT / "configs/ld_resolution1024_test.py"):
    """Load the RevealLayer pipeline and the transparent decoder (bfloat16) once."""
    from infer import initialize_pipeline, parse_config
    from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE

    paths = model_paths(models_dir)
    validate_model_paths(paths)
    os.environ.setdefault("REVEALLAYER_VIT32_DIR", str(paths.vit32.parent))
    args = SimpleNamespace(pretrained_model_name_or_path=str(paths.flux), ckpt_dir=str(paths.reveallayer),
                           transp_vae_ckpt=str(paths.transparent_decoder), extra_lora_dir=None, gpu_id=gpu_id,
                           max_layer=max_layer)
    device = torch.device("cuda", index=gpu_id)
    pipeline = initialize_pipeline(parse_config(str(cfg_path)), args)
    decoder = CustomVAE()
    decoder.load_state_dict(torch.load(args.transp_vae_ckpt, map_location="cpu"), strict=False)
    decoder = decoder.to(device=device, dtype=torch.bfloat16).eval()
    return pipeline, decoder, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_image_and_boxes(
    image: Image.Image,
    raw_boxes: list[list[float]],
    *,
    max_layers: int = 12,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], ImageGeometry]:
    """Apply the resize/crop/BBox geometry of upstream ``infer.py`` (1024 resolution, 16-pixel grid)."""
    from infer import (
        _transform_image_consistent_crop,
        adjust_coordinate,
        filter_and_align_bboxes,
        resize_with_bbox_edge,
    )

    image = image.convert("RGB")
    original_width, original_height = image.size
    (content_height, content_width), (target_height, target_width), (offset_y, offset_x), _ = (
        resize_with_bbox_edge(image, [], resolution=1024)
    )
    aligned_boxes = filter_and_align_bboxes(
        raw_boxes, original_width, original_height, content_width, content_height,
        offset_x, offset_y, target_width, target_height,
    )
    resized = _transform_image_consistent_crop(
        image, content_width, content_height, target_width, target_height, offset_x, offset_y,
    )
    boxes = [[0, 0, target_width, target_height]] * 2 + aligned_boxes
    final_boxes = [
        (
            adjust_coordinate(box[0], "floor", max_val=target_width),
            adjust_coordinate(box[1], "floor", max_val=target_height),
            adjust_coordinate(box[2], "ceil", max_val=target_width),
            adjust_coordinate(box[3], "ceil", max_val=target_height),
        )
        for box in boxes
    ][:max_layers]
    return resized, final_boxes, ImageGeometry(
        original_size=image.size,
        content_size=(content_width, content_height),
        target_size=(target_width, target_height),
        offset=(offset_x, offset_y),
    )


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return transform(image.convert("RGB")).to(
        device=device, dtype=torch.bfloat16, memory_format=torch.contiguous_format
    ).unsqueeze(0)


def encode_latent(pipeline, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Scaled latent (posterior mode, float32) of ``image``."""
    vae = pipeline.vae
    latent = vae.encode(image_to_tensor(image, device)).latent_dist.mode()
    return ((latent - vae.config.shift_factor) * vae.config.scaling_factor).float()


def grey_latent(pipeline, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Latent of a mid-grey image (pixel 0 in [-1, 1]): RevealLayer's transparent colour."""
    vae = pipeline.vae
    pixels = torch.zeros((1, 3, height, width), device=device, dtype=vae.dtype)
    latent = vae.encode(pixels).latent_dist.mode()
    return ((latent - vae.config.shift_factor) * vae.config.scaling_factor).float()


def latent_to_image(pipeline, latent: torch.Tensor) -> Image.Image:
    latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latent.to(pipeline.vae.dtype), return_dict=False)[0]
    array = ((image[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")
