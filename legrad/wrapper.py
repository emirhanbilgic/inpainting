import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
from open_clip.transformer import VisionTransformer
from open_clip.timm_model import TimmModel
from einops import rearrange

from .utils import hooked_resblock_forward, \
    hooked_attention_forward, \
    hooked_resblock_timm_forward, \
    hooked_attentional_pooler_timm_forward, \
    vit_dynamic_size_forward, \
    min_max, \
    hooked_torch_multi_head_attention_forward


class LeWrapper(nn.Module):
    """
    Wrapper around OpenCLIP to add LeGrad to OpenCLIP's model while keep all the functionalities of the original model.
    """

    def __init__(self, model, layer_index=-2):
        super(LeWrapper, self).__init__()
        # ------------ copy of model's attributes and methods ------------
        for attr in dir(model):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(model, attr))

        # ------------ activate hooks & gradient ------------
        self._activate_hooks(layer_index=layer_index)

    def _activate_hooks(self, layer_index):
        # ------------ identify model's type ------------
        print('Activating necessary hooks and gradients ....')
        if isinstance(self.visual, VisionTransformer):
            # --- Activate dynamic image size ---
            if not hasattr(self.visual, 'original_pos_embed'):
                self.visual.original_pos_embed = self.visual.positional_embedding.clone()
            self.visual.forward = types.MethodType(vit_dynamic_size_forward, self.visual)
            # Get patch size
            self.patch_size = self.visual.patch_size[0]
            # Get starting depth (in case of negative layer_index)
            self.starting_depth = layer_index if layer_index >= 0 else len(
                self.visual.transformer.resblocks) + layer_index

            if self.visual.attn_pool is None:
                self.model_type = 'clip'
                self._activate_self_attention_hooks()
            else:
                self.model_type = 'coca'
                self._activate_att_pool_hooks(layer_index=layer_index)

        elif isinstance(self.visual, TimmModel):
            # --- Activate dynamic image size ---
            self.visual.trunk.dynamic_img_size = True
            self.visual.trunk.patch_embed.dynamic_img_size = True
            self.visual.trunk.patch_embed.strict_img_size = False
            self.visual.trunk.patch_embed.flatten = False
            self.visual.trunk.patch_embed.output_fmt = 'NHWC'
            self.model_type = 'timm_siglip'
            # --- Get patch size ---
            self.patch_size = self.visual.trunk.patch_embed.patch_size[0]
            # --- Get starting depth (in case of negative layer_index) ---
            self.starting_depth = layer_index if layer_index >= 0 else len(
                self.visual.trunk.blocks) + layer_index
            self._activate_timm_attn_pool_hooks(layer_index=layer_index)
        else:
            raise ValueError(
                "Model currently not supported, see legrad.list_pretrained() for a list of available models")
        print('Hooks and gradients activated!')

    def _activate_self_attention_hooks(self):
        # ---------- Apply Hooks + Activate/Deactivate gradients ----------
        # Necessary steps to get intermediate representations
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.transformer.resblocks'):
                # get the depth
                depth = int(name.split('visual.transformer.resblocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers ---
        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            self.visual.transformer.resblocks[layer].attn.forward = types.MethodType(hooked_torch_multi_head_attention_forward,
                                                                                     self.visual.transformer.resblocks[
                                                                                         layer].attn)
            self.visual.transformer.resblocks[layer].forward = types.MethodType(hooked_resblock_forward,
                                                                                self.visual.transformer.resblocks[
                                                                                    layer])

    def _activate_att_pool_hooks(self, layer_index):
        # ---------- Apply Hooks + Activate/Deactivate gradients ----------
        # Necessary steps to get intermediate representations
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.transformer.resblocks'):
                # get the depth
                depth = int(name.split('visual.transformer.resblocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers ---
        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            self.visual.transformer.resblocks[layer].forward = types.MethodType(hooked_resblock_forward,
                                                                                self.visual.transformer.resblocks[
                                                                                    layer])
        # --- Apply hook on the attentional pooler ---
        self.visual.attn_pool.attn.forward = types.MethodType(hooked_torch_multi_head_attention_forward,
                                                              self.visual.attn_pool.attn)

    def _activate_timm_attn_pool_hooks(self, layer_index):
        # --- Deactivate gradient for module that don't need it ---

        # --- Deactivate gradient for module that don't need it ---
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith('visual.trunk.attn_pool'):
                param.requires_grad = True
            if name.startswith('visual.trunk.blocks'):
                # get the depth
                depth = int(name.split('visual.trunk.blocks.')[-1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True

        # --- Activate the hooks for the specific layers by modifying the block's forward ---
        for layer in range(self.starting_depth, len(self.visual.trunk.blocks)):
            self.visual.trunk.blocks[layer].forward = types.MethodType(hooked_resblock_timm_forward,
                                                                       self.visual.trunk.blocks[layer])

        self.visual.trunk.attn_pool.forward = types.MethodType(hooked_attentional_pooler_timm_forward,
                                                               self.visual.trunk.attn_pool)

    def compute_legrad(self, text_embedding, image=None, apply_correction=True):
        if 'clip' in self.model_type:
            return self.compute_legrad_clip(text_embedding, image)
        elif 'siglip' in self.model_type:
            return self.compute_legrad_siglip(text_embedding, image, apply_correction=apply_correction)
        elif 'coca' in self.model_type:
            return self.compute_legrad_coca(text_embedding, image)

    # -------------------------------------------------------------------------
    # Grad-CAM style explainability for CLIP
    # -------------------------------------------------------------------------
    def compute_gradcam(self, text_embedding, image=None, target_layer=None):
        """
        Compute Grad-CAM-style heatmaps for CLIP models.

        This mirrors the LeGrad API and returns a tensor of shape [1, P, H, W]
        (P = number of prompts), min-max normalised per prompt.

        Args:
            text_embedding: Tensor of shape [P, d], text embeddings (will be L2-normalised inside).
            image: image tensor of shape [1, 3, H, W] or [B, 3, H, W].
            target_layer: Index of the transformer block to use as the Grad-CAM target.
                If None, we pick an empirically good default (e.g., 8 for ViT-B/16).
        """
        if 'clip' in self.model_type:
            return self.compute_gradcam_clip(text_embedding, image, target_layer=target_layer)
        else:
            raise NotImplementedError(
                f"compute_gradcam is currently implemented only for CLIP VisionTransformer backbones, "
                f"got model_type={self.model_type}"
            )

    def compute_legrad_clip(self, text_embedding, image=None):
        num_prompts = text_embedding.shape[0]
        if image is not None:
            image = image.repeat(num_prompts, 1, 1, 1)
            _ = self.encode_image(image)
            # Save spatial size for later upsampling
            H_img, W_img = image.shape[-2], image.shape[-1]

        blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())

        image_features_list = []

        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
            intermediate_feat = self.visual.ln_post(intermediate_feat.mean(dim=0)) @ self.visual.proj
            intermediate_feat = F.normalize(intermediate_feat, dim=-1)
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        if hasattr(self.visual, 'last_grid_size'):
            w, h = self.visual.last_grid_size
        else:
            w = h = int(math.sqrt(num_tokens))

        # ----- Get explainability map
        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
            self.visual.zero_grad()
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
            one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
            one_hot = torch.sum(one_hot * sim)

            attn_map = blocks_list[self.starting_depth + layer].attn.attention_maps  # [b, num_heads, N, N]

            # -------- Get explainability map --------
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
                0]  # [batch_size * num_heads, N, N]
            grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)  # separate batch and attn heads
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # average attn over [CLS] + patch tokens
            expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
            # Upsample to either exact image size (preferred) or patch-aligned grid
            if image is not None:
                expl_map = F.interpolate(expl_map, size=(H_img, W_img), mode='bilinear',
                                         align_corners=False)  # [B, 1, H, W]
            else:
                expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
            accum_expl_map += expl_map

        # Min-Max Norm
        accum_expl_map = min_max(accum_expl_map)
        return accum_expl_map

    def compute_gradcam_clip(self, text_embedding, image=None, target_layer=None, allow_fallback=False):
        """
        Grad-CAM for CLIP ViT following the paper's formula:
        
        w = (1/n) * Σ_{i=0}^n (∂s/∂z_i^l)  ∈ R^d
        Ê_GradCAM = ((1/d) * Σ_{k=1}^d (w_k * Z^l_{1:,k}))^+  ∈ R^n
        
        Key insight from the paper:
        - Weights w are computed by averaging gradients across ALL tokens (including CLS)
        - The CAM is computed using only patch token activations (excluding CLS)
        
        Args:
            text_embedding: [P, d] text embeddings (will be L2-normalised inside).
            image: image tensor, [1, 3, H, W]; it will be repeated P times.
            target_layer: which transformer block to use (int). If None, we choose a model-aware default:
                          for ViT-B/16 (12 layers) we use layer 8 (0-indexed); otherwise mid-late block.
            allow_fallback: if True, when the selected layer yields a zero CAM, aggregate Grad-CAM across layers.
        Returns:
            Tensor of shape [1, P, H, W] with values in [0, 1].
        """
        if image is None:
            raise ValueError("compute_gradcam_clip requires an image tensor (cannot be None).")

        num_prompts = text_embedding.shape[0]

        # L2-normalise both text and image embeddings for cosine-style similarity
        text_embedding = F.normalize(text_embedding, dim=-1)

        # Select target transformer block
        blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())
        if target_layer is None:
            # Pick an empirically good default
            num_blocks = len(blocks_list)
            # Paper: "layer 8 of ViT-B/16 yields optimal results" (using 0-indexed)
            target_layer = 8 if num_blocks == 12 and getattr(self, "patch_size", None) == 16 else max(self.starting_depth, num_blocks // 2)
        if target_layer < 0 or target_layer >= len(blocks_list):
            raise ValueError(f"Invalid target_layer index {target_layer} for Grad-CAM (num_blocks={len(blocks_list)}).")

        target_block = blocks_list[target_layer]

        # Containers for forward activations and backward gradients
        activations = {}
        gradients = {}

        def _forward_hook(module, inp, out):
            # out: [L, B, C] where L = num_tokens (including CLS)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
            # Store patch tokens only for activations (excluding CLS at index 0)
            # Paper formula: Z^l_{1:,k} means patch tokens only
            activations["patch_tokens"] = x[1:].permute(1, 0, 2)  # [B, L_p, C]
            
            # Capture gradients w.r.t. the FULL output tensor (including CLS)
            def _tensor_grad_hook(grad):
                # grad has same shape as out: [L, B, C]
                # Paper formula: w = (1/n) * Σ_{i=0}^n (∂s/∂z_i^l) includes ALL tokens
                gradients["all_tokens"] = grad.permute(1, 0, 2)  # [B, L, C] including CLS
            x.register_hook(_tensor_grad_hook)

        # Register hooks
        h_fwd = target_block.register_forward_hook(_forward_hook)

        try:
            # Repeat image per prompt so that each sample corresponds to one prompt
            image_rep = image.repeat(num_prompts, 1, 1, 1)  # [P, 3, H, W]
            # Ensure gradients can flow to intermediate tensors
            image_rep = image_rep.clone().detach().requires_grad_(True)

            # Forward pass to populate hooks and get image embeddings
            image_emb = self.encode_image(image_rep)  # [P, d_img]
            image_emb = F.normalize(image_emb, dim=-1)  # [P, d_img]

            # Similarity scores per (prompt, image) pair: use per-prompt diagonal
            sims = torch.sum(text_embedding * image_emb, dim=-1)  # [P]
            score = sims.sum()

            # Backward pass: populates gradients
            self.zero_grad()
            score.backward(retain_graph=False)

            if "patch_tokens" not in activations or "all_tokens" not in gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            feats_patches = activations["patch_tokens"]   # [P, L_p, C] - patch tokens only
            grads_all = gradients["all_tokens"]           # [P, L, C] - ALL tokens including CLS

            # Paper formula: w = (1/n) * Σ_{i=0}^n (∂s/∂z_i^l)
            # Global-average-pool gradients over ALL tokens (including CLS) to get per-channel weights
            weights = grads_all.mean(dim=1)    # [P, C]

            # Paper formula: Ê = (1/d) * Σ_k (w_k * Z^l_{1:,k})
            # Weighted combination of patch token activations across channels
            cam = torch.einsum("plc,pc->pl", feats_patches, weights)  # [P, L_p]

            # ReLU as in standard Grad-CAM (the ^+ in the paper formula)
            cam = F.relu(cam)
        finally:
            # Remove hooks to avoid leaking references
            h_fwd.remove()

        # Reshape tokens to spatial grid
        num_tokens = feats_patches.shape[1]
        if hasattr(self.visual, 'last_grid_size'):
            h, w = self.visual.last_grid_size
        else:
            h = w = int(math.sqrt(num_tokens))
        cam = cam.view(num_prompts, h, w)  # [P, H_t, W_t]
        cam = cam.unsqueeze(0)  # [1, P, H_t, W_t]

        # If this layer yields a degenerate CAM (all zeros), optionally fall back to aggregating layers
        degenerate = (cam.detach().abs().max() == 0)
        if degenerate and allow_fallback:
            # Re-run a forward pass with hooks on all blocks to capture activations+grads,
            # then aggregate Grad-CAM across layers using the same formula.
            acts_by_layer = {}
            grads_by_layer = {}
            hook_handles = []

            def make_fwd_hook(layer_idx):
                def _fwd(module, inp, out):
                    x = out[0] if isinstance(out, tuple) else out  # [L, P, C]
                    # Store patch tokens only for activations
                    acts_by_layer[layer_idx] = x[1:].permute(1, 0, 2)  # [P, L_p, C]
                    # Capture gradients for ALL tokens (including CLS)
                    def _grad_hook(grad):
                        grads_by_layer[layer_idx] = grad.permute(1, 0, 2)  # [P, L, C]
                    x.register_hook(_grad_hook)
                return _fwd

            # Register hooks for all layers we care about
            for li in range(self.starting_depth, len(blocks_list)):
                h = blocks_list[li].register_forward_hook(make_fwd_hook(li))
                hook_handles.append(h)

            try:
                image_rep = image.repeat(num_prompts, 1, 1, 1)
                image_emb2 = self.encode_image(image_rep)                # [P, d_img]
                image_emb2 = F.normalize(image_emb2, dim=-1)
                sims2 = torch.sum(text_embedding * image_emb2, dim=-1)   # [P]
                score2 = sims2.sum()
                self.zero_grad()
                score2.backward(retain_graph=False)
            finally:
                for h in hook_handles:
                    h.remove()

            # Aggregate per-layer Grad-CAM
            accum = None
            # grid size from last forward
            if hasattr(self.visual, 'last_grid_size'):
                h_g, w_g = self.visual.last_grid_size
            else:
                # infer from any captured layer
                sample_layer = next(iter(acts_by_layer)) if len(acts_by_layer) else None
                if sample_layer is not None:
                    Lp = acts_by_layer[sample_layer].shape[1]
                    h_g = w_g = int(math.sqrt(Lp))
                else:
                    h_g, w_g = h, w

            for li in range(self.starting_depth, len(blocks_list)):
                A = acts_by_layer.get(li, None)   # [P, L_p, C] - patch tokens only
                G = grads_by_layer.get(li, None)  # [P, L, C] - ALL tokens including CLS
                if A is None or G is None:
                    continue
                # Average gradients over ALL tokens (including CLS) for weights
                wts = G.mean(dim=1)               # [P, C]
                cam_l = torch.einsum("plc,pc->pl", A, wts)  # [P, L_p]
                cam_l = F.relu(cam_l)
                cam_l = cam_l.view(num_prompts, h_g, w_g)
                accum = cam_l if accum is None else (accum + cam_l)
            if accum is None:
                accum = torch.zeros_like(cam.view(num_prompts, h, w))
            cam = accum.unsqueeze(0)  # [1, P, H_t, W_t]

        # Upsample to the exact input image resolution
        H_img, W_img = image.shape[-2], image.shape[-1]
        cam = F.interpolate(cam, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # Min-max normalisation per prompt (handled per [B, P] by min_max)
        cam = min_max(cam)
        return cam

    def compute_legrad_coca(self, text_embedding, image=None):
        if image is not None:
            _ = self.encode_image(image)
            H_img, W_img = image.shape[-2], image.shape[-1]

        blocks_list = list(dict(self.visual.transformer.resblocks.named_children()).values())

        image_features_list = []

        for layer in range(self.starting_depth, len(self.visual.transformer.resblocks)):
            intermediate_feat = self.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
            intermediate_feat = intermediate_feat.permute(1, 0, 2)  # [batch, num_patch, dim]
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
        w = h = int(math.sqrt(num_tokens))

        # ----- Get explainability map
        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
            self.visual.zero_grad()
            # --- Apply attn_pool ---
            image_embedding = self.visual.attn_pool(img_feat)[:,
                              0]  # we keep only the first pooled token as it is only this one trained with the contrastive loss
            image_embedding = image_embedding @ self.visual.proj

            sim = text_embedding @ image_embedding.transpose(-1, -2)  # [1, 1]
            one_hot = torch.sum(sim)

            attn_map = self.visual.attn_pool.attn.attention_maps  # [num_heads, num_latent, num_patch]

            # -------- Get explainability map --------
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
                0]  # [num_heads, num_latent, num_patch]
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=0)[0, 1:]  # average attn over heads + select first latent
            expl_map = rearrange(image_relevance, '(w h) -> 1 1 w h', w=w, h=h)
            if image is not None:
                expl_map = F.interpolate(expl_map, size=(H_img, W_img), mode='bilinear',
                                         align_corners=False)  # [B, 1, H, W]
            else:
                expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]
            accum_expl_map += expl_map

        # Min-Max Norm
        accum_expl_map = (accum_expl_map - accum_expl_map.min()) / (accum_expl_map.max() - accum_expl_map.min())
        return accum_expl_map

    def _init_empty_embedding(self):
        if not hasattr(self, 'empty_embedding'):
            # For the moment only SigLIP is supported & they all have the same tokenizer
            _tok = open_clip.get_tokenizer(model_name='ViT-B-16-SigLIP')
            empty_text = _tok(['a photo of a']).to(self.logit_scale.data.device)
            empty_embedding = self.encode_text(empty_text)
            empty_embedding = F.normalize(empty_embedding, dim=-1)
            self.empty_embedding = empty_embedding.t()

    def compute_legrad_siglip(self, text_embedding, image=None, apply_correction=True, correction_threshold=0.8):
        # --- Forward CLIP ---
        blocks_list = list(dict(self.visual.trunk.blocks.named_children()).values())
        if image is not None:
            _ = self.encode_image(image)  # [bs, num_patch, dim] bs=num_masks
            H_img, W_img = image.shape[-2], image.shape[-1]

        image_features_list = []
        for blk in blocks_list[self.starting_depth:]:
            intermediate_feat = blk.feat_post_mlp
            image_features_list.append(intermediate_feat)

        num_tokens = blocks_list[-1].feat_post_mlp.shape[1]
        w = h = int(math.sqrt(num_tokens))

        if apply_correction:
            self._init_empty_embedding()
            accum_expl_map_empty = 0

        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
            self.zero_grad()
            pooled_feat = self.visual.trunk.attn_pool(img_feat)
            pooled_feat = F.normalize(pooled_feat, dim=-1)
            # -------- Get explainability map --------
            sim = text_embedding @ pooled_feat.transpose(-1, -2)  # [num_mask, num_mask]
            one_hot = torch.sum(sim)
            grad = torch.autograd.grad(one_hot, [self.visual.trunk.attn_pool.attn_probs], retain_graph=True,
                                       create_graph=True)[0]
            grad = torch.clamp(grad, min=0.)

            image_relevance = grad.mean(dim=1)[:, 0]  # average attn over [CLS] + patch tokens
            expl_map = rearrange(image_relevance, 'b (w h) -> b 1 w h', w=w, h=h)
            accum_expl_map += expl_map

            if apply_correction:
                # -------- Get empty explainability map --------
                sim_empty = pooled_feat @ self.empty_embedding
                one_hot_empty = torch.sum(sim_empty)
                grad_empty = \
                    torch.autograd.grad(one_hot_empty, [self.visual.trunk.attn_pool.attn_probs], retain_graph=True,
                                        create_graph=True)[0]
                grad_empty = torch.clamp(grad_empty, min=0.)

                image_relevance_empty = grad_empty.mean(dim=1)[:, 0]  # average attn over heads + select query's row
                expl_map_empty = rearrange(image_relevance_empty, 'b (w h) -> b 1 w h', w=w, h=h)
                accum_expl_map_empty += expl_map_empty

        if apply_correction:
            heatmap_empty = min_max(accum_expl_map_empty)
            accum_expl_map[heatmap_empty > correction_threshold] = 0

        Res = min_max(accum_expl_map)
        if image is not None:
            Res = F.interpolate(Res, size=(H_img, W_img), mode='bilinear',
                                align_corners=False)  # [B, 1, H, W]
        else:
            Res = F.interpolate(Res, scale_factor=self.patch_size, mode='bilinear')  # [B, 1, H, W]

        return Res


class LePreprocess(nn.Module):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    """

    def __init__(self, preprocess, image_size):
        super(LePreprocess, self).__init__()
        self.transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                preprocess.transforms[-3],
                preprocess.transforms[-2],
                preprocess.transforms[-1],
            ]
        )

    def forward(self, image):
        return self.transform(image)
