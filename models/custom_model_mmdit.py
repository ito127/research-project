import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple
import  os
from accelerate.utils import set_module_tensor_to_device
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)

import torch.nn.functional as F
from einops import rearrange, repeat
from models.adapter import HybridRefiner
import torch.nn.utils.parametrize as parametrize

class Clamp(nn.Module):
    """
    A module that clamps the input tensor to a specified range.
    """
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)
    def right_inverse(self, y: torch.Tensor) -> torch.Tensor:
        # This allows the initial parameter to be set correctly
        return y

class CustomFluxTransformer2DModel(FluxTransformer2DModel):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        max_layer_num: int = 12,
    ):
        super(FluxTransformer2DModel, self).__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.max_layer_num = max_layer_num

        layer_pe_value = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(
                1, self.max_layer_num, 1, 1, self.inner_dim,
            )),
            mean=0.0, std=0.02, a=-2.0, b=2.0,
        ).data.detach()
        self.layer_pe = nn.Parameter(layer_pe_value)
        set_module_tensor_to_device(
            self, 
            'layer_pe', 
            device='cpu',
            value=layer_pe_value,
            dtype=layer_pe_value.dtype,
        )

        self.refiner = HybridRefiner(
            feature_dim=self.inner_dim,
            pixel_dim=16,
            time_emb_dim=self.inner_dim,
            attn_hidden=1024,
            num_layers=1,
            patch_size=2,
            )
        for param in self.refiner.parameters():
            param.requires_grad = True  

    @classmethod
    def from_pretrained(cls, *args, **kwarg):
        model = super().from_pretrained(*args, **kwarg)
        for name, para in model.named_parameters():
            if name != 'layer_pe':
                device = para.device
                break
        model.layer_pe.to(device)
        return model

    def crop_each_layer(self, hidden_states, list_layer_box, factor):
        """
            hidden_states: [1, n_layers, h, w, inner_dim]
            list_layer_box: List, length=n_layers, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        token_list = []
        for layer_idx in range(hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // factor, y1 // factor, x2 // factor, y2 // factor
                layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2, :]
                bs, h, w, c = layer_token.shape
                layer_token = layer_token.reshape(bs, -1, c)
                # mask
                #layer_token_mask = union_mask(layer_token, mask_layer)

                token_list.append(layer_token)
        result = torch.cat(token_list, dim=1)
        return result

    def get_overlapMask(self, is_overlap_region, bg_overlap, list_layer_box, factor):
        B, H, W = is_overlap_region.shape
        segmented_flags_list = []
        # full_overlap = torch.zeros_like(bg_overlap)
        for layer_idx in range(0, len(list_layer_box)):
            if layer_idx == 0:
                x1, y1, x2, y2 = list_layer_box[layer_idx] 
                x1, y1, x2, y2 = x1 // factor, y1 // factor, x2 // factor, y2 // factor            
                region_mask = bg_overlap[:, y1:y2, x1:x2]
                bs, region_h, region_w = region_mask.shape
                layer_flags = region_mask.float().view(bs, -1, 1) # Shape: [B, region_h*region_w, 1]
                segmented_flags_list.append(layer_flags)
            elif layer_idx == 1:
                x1, y1, x2, y2 = list_layer_box[layer_idx] 
                x1, y1, x2, y2 = x1 // factor, y1 // factor, x2 // factor, y2 // factor            
                region_mask = bg_overlap[:, y1:y2, x1:x2]
                bs, region_h, region_w = region_mask.shape
                layer_flags = region_mask.float().view(bs, -1, 1) # Shape: [B, region_h*region_w, 1]
                segmented_flags_list.append(layer_flags)
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx] 
                x1, y1, x2, y2 = x1 // factor, y1 // factor, x2 // factor, y2 // factor            
                region_mask = is_overlap_region[:, y1:y2, x1:x2]
                bs, region_h, region_w = region_mask.shape
                layer_flags = region_mask.float().view(bs, -1, 1) # Shape: [B, region_h*region_w, 1]
                
                segmented_flags_list.append(layer_flags)

        result = torch.cat(segmented_flags_list, dim=1)      
        return result

    def fill_in_processed_tokens(self, hidden_states, full_hidden_states, list_layer_box):
        """
            hidden_states: [1, h1xw1 + h2xw2 + ... + hlxwl , inner_dim]
            full_hidden_states: [1, n_layers, h, w, inner_dim]
            list_layer_box: List, length=n_layers, each element is a Tuple of 4 elements (x1, y1, x2, y2)
        """
        used_token_len = 0
        bs = hidden_states.shape[0]
        for layer_idx in range(full_hidden_states.shape[1]):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                full_hidden_states[:, layer_idx, y1:y2, x1:x2, :] = hidden_states[:, used_token_len: used_token_len + (y2-y1) * (x2-x1), :].reshape(bs, y2-y1, x2-x1, -1)
                used_token_len = used_token_len + (y2-y1) * (x2-x1)
        return full_hidden_states

    def build_additive_spatial_roi_mask_v2(self, list_layer_box, height, width, t_text_length, device, dtype, self_value=0, enhance_value=0):
        latent_H, latent_W = height // 2, width // 2 
        
        token_positions = []
        token_layer_ids = []
        token_rois = []
        
        for layer_idx, box in enumerate(list_layer_box):
            if box is None: 
                continue

            x1, y1, x2, y2 = box
            lx1, ly1 = max(0, x1 // 16), max(0, y1 // 16)
            lx2, ly2 = min(latent_W, x2 // 16), min(latent_H, y2 // 16)
            h, w = max(1, ly2 - ly1), max(1, lx2 - lx1)
            
            for dy in range(h):
                for dx in range(w):
                    token_positions.append((lx1 + dx, ly1 + dy))
                    token_layer_ids.append(layer_idx)
                    token_rois.append((lx1, ly1, lx2, ly2))
                    
        t_img_length = len(token_positions)
        if t_img_length == 0:
            return torch.zeros(t_text_length, t_text_length, dtype=dtype, device=device)

        xs = torch.tensor([p[0] for p in token_positions], device=device) # Shape: [T_img]
        ys = torch.tensor([p[1] for p in token_positions], device=device) # Shape: [T_img]
        
        layer_ids = torch.tensor(token_layer_ids, device=device) # Shape: [T_img]
        
        roi_x1 = torch.tensor([r[0] for r in token_rois], device=device) # Shape: [T_img]
        roi_y1 = torch.tensor([r[1] for r in token_rois], device=device) # Shape: [T_img]
        roi_x2 = torch.tensor([r[2] for r in token_rois], device=device) # Shape: [T_img]
        roi_y2 = torch.tensor([r[3] for r in token_rois], device=device) # Shape: [T_img]

        t_total = t_text_length + t_img_length
        
        full_mask = torch.full((t_total, t_total), -1e9, dtype=dtype, device=device)
        
        text_slice = slice(0, t_text_length)
        img_slice = slice(t_text_length, t_total)
        
        full_mask[text_slice, text_slice] = enhance_value
        full_mask[text_slice, img_slice] = enhance_value
        full_mask[img_slice, text_slice] = enhance_value

        q_layer = layer_ids.unsqueeze(1) 
        k_layer = layer_ids.unsqueeze(0) 
        
        img_mask_view = full_mask[img_slice, img_slice]

        is_same_layer = (q_layer == k_layer) # Shape: [T_img, T_img]
        
        img_mask_view[is_same_layer] = self_value

        is_bg_query   = (q_layer == 1)
        is_not_bg_key = (k_layer != 1)
        
        bg_sees_others = is_bg_query & is_not_bg_key # Shape: [T_img, T_img]
        
        img_mask_view[bg_sees_others] = enhance_value


        is_fg_query   = (q_layer >= 2)
        is_full_key   = (k_layer == 0) 
        
        q_x1 = roi_x1.unsqueeze(1)
        q_x2 = roi_x2.unsqueeze(1)
        q_y1 = roi_y1.unsqueeze(1)
        q_y2 = roi_y2.unsqueeze(1)
        
        k_x_pos = xs.unsqueeze(0)
        k_y_pos = ys.unsqueeze(0)

        is_spatial_inside = (k_x_pos >= q_x1) & (k_x_pos < q_x2) & \
                            (k_y_pos >= q_y1) & (k_y_pos < q_y2)
        
        fg_sees_full_spatial = is_fg_query & is_full_key & is_spatial_inside
        
        img_mask_view[fg_sees_full_spatial] = enhance_value

        return full_mask

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, L, C, H, W]`` FLUX latents to patch features."""
        bs, n_layers, channel_latent, height, width = latents.shape
        latents = latents.view(bs, n_layers, channel_latent, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        return latents.reshape(bs, n_layers, height // 2, width // 2, channel_latent * 4)

    @staticmethod
    def crop_token_maps(token_maps: torch.Tensor, list_layer_box: List[Tuple]) -> torch.Tensor:
        """Crop per-layer token maps ``[B, L, h, w]`` into the model's image-token order, as ``[B, T_img, 1]``."""
        token_masks = []
        for layer_idx, box in enumerate(list_layer_box):
            if box is None:
                continue
            x1, y1, x2, y2 = (v // 16 for v in box)
            token_mask = token_maps[:, layer_idx, y1:y2, x1:x2]
            token_masks.append(token_mask.reshape(token_mask.shape[0], -1, 1))
        return torch.cat(token_masks, dim=1).to(dtype=torch.bool)

    @staticmethod
    def reference_image_ids(box, img_ids):
        """Positions of a reference crop: layer id 0 (as the full-image layer) and the BBox's token grid."""
        x1, y1, x2, y2 = (v // 16 for v in box)
        ys, xs = torch.meshgrid(torch.arange(y1, y2), torch.arange(x1, x2), indexing="ij")
        ids = torch.zeros((ys.numel(), 3), dtype=img_ids.dtype, device=img_ids.device)
        ids[:, 1] = ys.flatten().to(ids.device)
        ids[:, 2] = xs.flatten().to(ids.device)
        return ids

    @staticmethod
    def extend_mask_with_foreground_reference(full_mask, list_layer_box, t_text_length, reference_box,
                                              enhance_value=0):
        """Append the foreground-reference tokens (a second full-image layer) to the attention mask.

        - object-layer queries no longer see the full-image layer (index 0); instead they see the
          reference tokens inside their own BBox, as they saw layer 0 before;
        - reference queries see the reference tokens and the text tokens;
        - text, full-image and background queries never see the reference tokens.
        """
        device = full_mask.device
        layer_ids, rois = [], []
        for layer_idx, box in enumerate(list_layer_box):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            n = (y2 // 16 - y1 // 16) * (x2 // 16 - x1 // 16)
            layer_ids += [layer_idx] * n
            rois += [(x1 // 16, y1 // 16, x2 // 16, y2 // 16)] * n
        layer_ids = torch.tensor(layer_ids, device=device)
        rois = torch.tensor(rois, device=device)
        t_old = full_mask.shape[0]
        if t_old != t_text_length + layer_ids.numel():
            raise RuntimeError("Attention mask does not match the layer boxes.")
        x1, y1, x2, y2 = (v // 16 for v in reference_box)
        ref_y, ref_x = torch.meshgrid(torch.arange(y1, y2, device=device), torch.arange(x1, x2, device=device),
                                      indexing="ij")
        ref_x, ref_y = ref_x.flatten(), ref_y.flatten()
        t_total = t_old + ref_x.numel()
        mask = torch.full((t_total, t_total), -1e9, dtype=full_mask.dtype, device=device)
        mask[:t_old, :t_old] = full_mask
        image = slice(t_text_length, t_old)
        reference = slice(t_old, t_total)
        object_query = layer_ids.unsqueeze(1) >= 2
        # Object layers: drop the full-image keys ...
        image_view = mask[image, image]
        image_view[object_query & (layer_ids == 0).unsqueeze(0)] = -1e9
        # ... and see the reference keys inside their BBox.
        inside = ((ref_x.unsqueeze(0) >= rois[:, 0:1]) & (ref_x.unsqueeze(0) < rois[:, 2:3])
                  & (ref_y.unsqueeze(0) >= rois[:, 1:2]) & (ref_y.unsqueeze(0) < rois[:, 3:4]))
        mask[image, reference][object_query & inside] = enhance_value
        # Reference queries: reference tokens and text only.
        mask[reference, reference] = enhance_value
        mask[reference, :t_text_length] = enhance_value
        return mask

    def block_tokens_from_background(self, attention_mask, blocked_token_maps, list_layer_box, t_text_length):
        """Hide the tokens marked in ``blocked_token_maps`` from the background-layer (layer 1) queries.

        Every other query row of ``attention_mask`` is unchanged.
        """
        blocked = self.crop_token_maps(blocked_token_maps, list_layer_box)
        if blocked.shape[0] != 1 and not torch.equal(blocked, blocked[:1].expand_as(blocked)):
            raise ValueError("blocked_token_maps must be the same for the whole batch.")
        layer_ids = []
        for layer_idx, box in enumerate(list_layer_box):
            if box is None:
                continue
            x1, y1, x2, y2 = (v // 16 for v in box)
            layer_ids += [layer_idx] * ((y2 - y1) * (x2 - x1))
        layer_ids = torch.tensor(layer_ids, device=attention_mask.device)
        rows = t_text_length + torch.nonzero(layer_ids == 1, as_tuple=False).flatten()
        cols = t_text_length + torch.nonzero(blocked[0, :, 0], as_tuple=False).flatten()
        blocked_mask = attention_mask.clone()
        if rows.numel() > 0 and cols.numel() > 0:
            blocked_mask[rows[:, None], cols[None, :]] = torch.finfo(attention_mask.dtype).min
        return blocked_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_data: torch.Tensor,
        split_sizes = None,
        list_layer_box: List[Tuple] = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        blocked_token_maps: Optional[torch.Tensor] = None,
        foreground_reference_latents: Optional[torch.Tensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """``blocked_token_maps`` ([batch, layers, H/2, W/2], bool): image tokens hidden from the background-layer
        queries in every MMDiT and single-stream block (see ``block_tokens_from_background``).
        ``foreground_reference_latents`` ([batch, 1, C, H, W], clean): an extra full-image layer seen only by the
        object layers (inside their BBoxes) in place of layer 0; see ``extend_mask_with_foreground_reference``.
        Its tokens are appended after the image tokens, use layer 0's layer embedding and positions, skip the
        refiner and are dropped from the output."""

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

        bs, n_layers, channel_latent, height, width = hidden_states.shape 

        # Patchify Noisy Latents
        hidden_states = self.patchify_latents(hidden_states)

        # [Flux Linear Layer]
        hidden_states = self.x_embedder(hidden_states) 
        full_hidden_states = torch.zeros_like(hidden_states) 
        
        # Layer PE
        layer_pe = self.layer_pe.view(1, self.max_layer_num, 1, 1, self.inner_dim) 
        hidden_states = hidden_states + layer_pe[:, :n_layers] 
        
        # Crop Sequence (Feature Space)
        hidden_states = self.crop_each_layer(hidden_states, list_layer_box, 16)
        num_layer_tokens = hidden_states.shape[1]

        reference_tokens = None
        if foreground_reference_latents is not None:
            if tuple(foreground_reference_latents.shape) != (bs, 1, channel_latent, height, width):
                raise ValueError("foreground_reference_latents must have shape [batch, 1, channels, height, width].")
            reference_states = self.x_embedder(self.patchify_latents(foreground_reference_latents.to(hidden_states.dtype)))
            reference_states = reference_states + layer_pe[:, :1]
            reference_tokens = self.crop_each_layer(reference_states, [list_layer_box[0]], 16)

        # Attention Mask
        T_text = encoder_hidden_states.shape[1]
        attention_mask = self.build_additive_spatial_roi_mask_v2(
            list_layer_box, height=height, width=width, t_text_length=T_text, device=hidden_states.device, dtype=hidden_states.dtype,
        )
        if reference_tokens is not None:
            attention_mask = self.extend_mask_with_foreground_reference(
                attention_mask, list_layer_box, T_text, list_layer_box[0])
            hidden_states = torch.cat([hidden_states, reference_tokens], dim=1)
        if blocked_token_maps is not None:
            if tuple(blocked_token_maps.shape) != (bs, n_layers, height // 2, width // 2):
                raise ValueError("blocked_token_maps must have shape [batch, layers, latent_height/2, latent_width/2].")
            attention_mask = self.block_tokens_from_background(attention_mask, blocked_token_maps, list_layer_box, T_text)

        # Embeddings...
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3: txt_ids = txt_ids[0]
        if img_ids.ndim == 3: img_ids = img_ids[0]
        if reference_tokens is not None:
            img_ids_all = torch.cat((img_ids, self.reference_image_ids(list_layer_box[0], img_ids)), dim=0)
        else:
            img_ids_all = img_ids
        ids = torch.cat((txt_ids, img_ids_all), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # --- Transformer Blocks Loop ---
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask = attention_mask,
                )

        clean_tokens = self.crop_each_layer(adapter_data, list_layer_box[1:], 16)  
        l0_len = split_sizes[0]
        ids_refiner = img_ids[l0_len:, :]
        image_rotary_emb_refiner = self.pos_embed(ids_refiner)

        delta = self.refiner.forward_feature_attn(
            hidden_states=hidden_states[:, l0_len:num_layer_tokens, :],
            clean_tokens =clean_tokens,
            temb=temb,
            split_sizes=split_sizes,
            image_rotary_emb=image_rotary_emb_refiner,
        )

        hidden_states_part2 = hidden_states[:, l0_len:num_layer_tokens, :] + delta
        hidden_states_final = torch.cat((hidden_states[:, :l0_len, :], hidden_states_part2,
                                         hidden_states[:, num_layer_tokens:, :]), dim=1)
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states_final], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask = attention_mask,
                )
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = hidden_states[:, :num_layer_tokens]

        hidden_states = self.fill_in_processed_tokens(hidden_states, full_hidden_states, list_layer_box) 
        hidden_states = hidden_states.view(bs, -1, self.inner_dim) 
        hidden_states = self.norm_out(hidden_states, temb) 
        hidden_states = self.proj_out(hidden_states) 

        # Unpatchify
        hidden_states = hidden_states.view(bs, n_layers, height//2, width//2, channel_latent, 2, 2)
        hidden_states = hidden_states.permute(0, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(bs, n_layers, channel_latent, height, width) 

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, )

        return Transformer2DModelOutput(sample=output)
