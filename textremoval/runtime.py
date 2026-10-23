"""Model loading, image geometry and latent helpers shared by the text-removal commands."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROMPT = "Decompose the image into foreground and background."
VAE_SCALE_FACTOR = 8


@dataclass(frozen=True)
class ModelPaths:
    flux: Path
    reveallayer: Path
    transparent_decoder: Path


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
    )


def validate_model_paths(paths: ModelPaths) -> None:
    missing = [path for path in paths.__dict__.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required model files/directories are missing (see README):\n{formatted}")


def load_models(models_dir: Path | str, *, gpu_id: int = 0, max_layer: int = 12):
    """Load the RevealLayer pipeline (FLUX.1-dev + RevealLayer LoRA, layer embedding and refiner) and the
    transparent decoder (bfloat16) once."""
    from diffusers import AutoencoderKL, FluxTransformer2DModel
    from diffusers.configuration_utils import FrozenDict
    from safetensors.torch import load_file

    from models.custom_model_mmdit import CustomFluxTransformer2DModel
    from models.custom_model_xvae import AutoencoderKLTransformerTraining as TransparentDecoder
    from models.custom_pipeline import CustomFluxPipelineCfg

    paths = model_paths(models_dir)
    validate_model_paths(paths)
    flux, ckpt = str(paths.flux), paths.reveallayer
    device = torch.device("cuda", index=gpu_id)

    # The FLUX transformer weights, copied into RevealLayer's transformer (+ layer embedding and refiner).
    transformer_orig = FluxTransformer2DModel.from_pretrained(flux, subfolder="transformer", torch_dtype=torch.bfloat16)
    config = dict(transformer_orig.config)
    config["_class_name"] = "CustomSD3Transformer2DModel"
    config["max_layer_num"] = max_layer
    transformer = CustomFluxTransformer2DModel.from_config(FrozenDict(config)).to(dtype=torch.bfloat16)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    del transformer_orig
    transformer.layer_pe.data.copy_(torch.load(ckpt / "layer_pe.pt", map_location="cpu"))
    refiner = torch.load(ckpt / "Refiner.pt", map_location="cpu")
    transformer.load_state_dict({f"refiner.{k}": v for k, v in refiner.items()}, strict=False)

    vae = AutoencoderKL.from_pretrained(flux, subfolder="vae", torch_dtype=torch.bfloat16)
    pipeline = CustomFluxPipelineCfg.from_pretrained(
        flux, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16).to(device)
    lora = load_file(ckpt / "pytorch_lora_weights.safetensors")
    pipeline.load_lora_weights({k.replace("transformer.module.", "transformer."): v for k, v in lora.items()},
                               adapter_name="layer")

    decoder = TransparentDecoder()
    decoder.load_state_dict(torch.load(paths.transparent_decoder, map_location="cpu"), strict=False)
    decoder = decoder.to(device=device, dtype=torch.bfloat16).eval()
    return pipeline, decoder, device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _snap(value: float, mode: str, max_val: int, k: int = 16) -> int:
    """Round to the 16-pixel token grid (floor or ceil) and clip to [0, max_val]."""
    rounded = (math.floor if mode == "floor" else math.ceil)(value / k) * k
    return max(0, min(rounded, max_val))


def prepare_image_and_boxes(
    image: Image.Image,
    raw_boxes: list[list[float]],
    *,
    max_layers: int = 12,
    resolution: int = 1024,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], ImageGeometry]:
    """RevealLayer's input geometry: resize the long side to ``resolution``, centre-crop to the 16-pixel grid and map
    the object boxes onto it (boxes less than half visible are dropped). Returns the image, the boxes
    ``[full image, background, objects...]`` and the geometry."""
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    scale = resolution / max(orig_w, orig_h)
    content_w, content_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    target_w, target_h = max(16, (content_w // 16) * 16), max(16, (content_h // 16) * 16)
    offset_x, offset_y = (content_w - target_w) // 2, (content_h - target_h) // 2
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS).crop(
        (offset_x, offset_y, offset_x + target_w, offset_y + target_h))

    object_boxes = []
    scale_w, scale_h = content_w / orig_w, content_h / orig_h
    for x1, y1, x2, y2 in raw_boxes:
        sx1, sy1, sx2, sy2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
        cx1, cy1 = max(0, min(sx1 - offset_x, target_w)), max(0, min(sy1 - offset_y, target_h))
        cx2, cy2 = max(0, min(sx2 - offset_x, target_w)), max(0, min(sy2 - offset_y, target_h))
        visible_w, visible_h, scaled_w, scaled_h = cx2 - cx1, cy2 - cy1, sx2 - sx1, sy2 - sy1
        if (visible_w > 1 and visible_h > 1 and scaled_w > 0 and visible_w / scaled_w >= 0.5
                and scaled_h > 0 and visible_h / scaled_h >= 0.5):
            object_boxes.append([_snap(cx1, "floor", target_w), _snap(cy1, "floor", target_h),
                                 _snap(cx2, "ceil", target_w), _snap(cy2, "ceil", target_h)])

    boxes = [[0, 0, target_w, target_h]] * 2 + object_boxes
    final_boxes = [(_snap(b[0], "floor", target_w), _snap(b[1], "floor", target_h),
                    _snap(b[2], "ceil", target_w), _snap(b[3], "ceil", target_h)) for b in boxes][:max_layers]
    return resized, final_boxes, ImageGeometry(
        original_size=(orig_w, orig_h),
        content_size=(content_w, content_h),
        target_size=(target_w, target_h),
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
