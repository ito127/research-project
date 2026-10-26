# Method: single-pass text removal with RevealLayer

RevealLayer decomposes an image into a background layer and RGBA object layers, one per input BBox, and recovers
the content hidden behind the objects. We use it as a scene-text remover: the text is the object, and the
background layer is the text-free image. The pretrained RevealLayer checkpoint is used as is; everything below
happens at inference time, in one denoising run.

Removing text with RevealLayer as is has two failure modes:

1. the background layer copies the text from the full-image layer it attends to, and
2. the background drifts away from the input outside the text region.

The method adds three things to the inference: separate inputs for the text layer and the background, a
background keep outside the text, and attention blocking inside it.

## Inputs

All images are brought to the RevealLayer resolution (1024 px on the long side, 16-pixel grid) with the upstream
geometry. `M` is the binary text mask (SCUT-EnsText: the largest 8-connected component for `largest_text`, the
whole mask for `all_text`), and `b` is its tight BBox.

| Input | Content | Used by |
|---|---|---|
| Image B | the original image | the full-image layer is made from it |
| Full-image layer (index 0) | image B with `M` (dilated by 2 px) filled from the colours just outside it, blurred more the further inside (σ from 0 at the edge to 12 px at 60 px) | layer 0, refiner condition of the background layer, keep target `z_src` (VAE posterior mode) |
| Image A | the original inside `M`; outside it, the colours of the mask boundary spread outwards with the same gradual blur | refiner condition of the text layer, and an extra full-image layer that only the text layer attends to (inside `b`) in place of layer 0 |
| Object layer (index 2) | BBox `b` | the text layer |

The text layer therefore sees an image that contains the text but none of its surroundings. The background sees an
image in which the text has already been roughly filled in.

## Per-step guidance

The pipeline runs 30 flow-matching steps (seed 43, guidance 1.0). At step `k`:

- **Text mask `M_k`.** At step 0, `M_0` is the input mask (on the 8-pixel latent grid). From step 1 on, `M_k` is the
  set of latent cells inside `b` where the text layer's clean estimate `x̂0 = z_k − σ_k v_k` differs from the
  transparent (mid-grey) latent by more than 0.9 (RMS over channels). The mask follows the text the model
  actually extracts.
- **Keep (velocity correction, after the transformer).** Outside `M_k` dilated by one latent cell, the background
  layer's clean estimate is replaced by `z_src`:
  `v' = v + (x̂0 − z_src) / σ_k`, so `z_k − σ_k v' = z_src` there. `z_src` is the latent of the full-image layer,
  which equals the original image outside `M` (dilated by 2 px). Inside `M` but outside the dilated `M_k`, it pulls
  the background to the filled colours instead of the text. Inside the dilated `M_k` the background is generated
  freely.
- **Attention blocking (before the transformer at step `k + 1`).** The full-image-layer tokens under `M_k`
  (16-pixel tokens) are hidden from every background-layer query in all 19 MMDiT and 38 single-stream blocks. To do
  this, the additive attention mask is set to −∞ for those query/key pairs. The background can no longer copy the
  text pixels. All other query rows are unchanged.

The output is the background layer (`bg_rgb.png`). The text layer (`text_layer_rgba.png`) comes out of the same
run.

## Where it lives in the code

| Part | File |
|---|---|
| Input preparation, the inference call | `textremoval/method.py` |
| `M_k`, keep and blocking callbacks | `textremoval/guidance.py` (`TextRemovalController`) |
| Boundary-colour fills | `textremoval/fill.py` |
| Pipeline arguments `blocked_token_maps_callback`, `velocity_correction_callback`, `foreground_adapter_image`, `foreground_reference_image` | `models/custom_pipeline.py` |
| Attention-mask blocking and the reference layer | `models/custom_model_mmdit.py` (`block_tokens_from_background`, `extend_mask_with_foreground_reference`) |

Every default of `TextRemovalConfig` (and of the matching `scripts/remove_text.py` options) is the reported
setting.
