import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d
from typing import Optional


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)
        not_mask = ~mask
        flat_x = x[not_mask, :]
        if flat_x.numel() > 0:
            flat_x = self.bn(flat_x)
            x[not_mask, :] = flat_x
        x = rearrange(x, "b h w d -> b d h w")
        return x


class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module with Guided Coverage.
    
    Implements the guided coverage attention from the paper:
    
    penalty_guided = penalty_base × (1 + α_s·Ŝ + α_r·max_k(R̂_k))
    
    Where:
    - penalty_base: Standard coverage-based penalty
    - Ŝ: Spatial map (stroke density) 
    - R̂_k: Relation maps (multi-label structural relations)
    - α_s, α_r: Scaling factors for spatial and relation guidance
    
    This guides the decoder to:
    1. Avoid revisiting regions with strokes (spatial guidance)
    2. Pay extra attention to structurally complex regions (relation guidance)
    """
    
    def __init__(
        self, 
        nhead: int, 
        dc: int, 
        cross_coverage: bool, 
        self_coverage: bool,
        # Guided coverage parameters
        use_guided_coverage: bool = False,
        alpha_spatial: float = 0.3,  # Weight for spatial guidance
        alpha_relation: float = 0.2,  # Weight for relation guidance
        # Legacy parameters (backward compatibility)
        use_spatial_guide: bool = False,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        
        # Guided coverage configuration
        self.use_guided_coverage = use_guided_coverage or use_spatial_guide
        self.alpha_spatial = alpha_spatial if use_guided_coverage else spatial_scale
        self.alpha_relation = alpha_relation

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self, 
        prev_attn: Tensor, 
        key_padding_mask: Tensor, 
        h: int, 
        curr_attn: Tensor,
        spatial_map: Optional[Tensor] = None,
        relation_map: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional guided coverage.
        
        Args:
            prev_attn: Previous attention weights [B*nhead, T, L]
            key_padding_mask: Padding mask [B, L]
            h: Height of feature map
            curr_attn: Current attention weights [B*nhead, T, L]
            spatial_map: Optional spatial map [B, 1, H, W]
            relation_map: Optional relation map [B, C, H, W] (multi-label)
            
        Returns:
            Coverage penalty [B*nhead, T, L]
        """
        t = curr_attn.shape[1]
        b = key_padding_mask.shape[0]
        w = key_padding_mask.shape[1] // h
        
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        # Cumulative coverage
        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        cov = self.conv(attns)
        cov = self.act(cov)
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        
        # Apply guided coverage if enabled
        if self.use_guided_coverage and (spatial_map is not None or relation_map is not None):
            cov = self._apply_guided_coverage(
                cov, spatial_map, relation_map, b, t, h, w
            )
        
        return cov
    
    def _apply_guided_coverage(
        self, 
        cov: Tensor, 
        spatial_map: Optional[Tensor],
        relation_map: Optional[Tensor],
        b: int, 
        t: int, 
        h: int, 
        w: int
    ) -> Tensor:
        """
        Apply guided coverage penalty.
        
        Implementation of temp-2.pdf formula:
        penalty_guided = penalty_base × (1 + α_s·Ŝ + α_r·max_k(R̂_k))
        
        Args:
            cov: Base coverage penalty [B*nhead, T, H*W]
            spatial_map: Spatial map [B, 1, H', W']
            relation_map: Relation map [B, C, H', W']
            b, t, h, w: Batch size, time steps, height, width
            
        Returns:
            Guided coverage penalty [B*nhead, T, H*W]
        """
        guidance_factor = torch.ones(b, h * w, device=cov.device, dtype=cov.dtype)
        
        # Spatial guidance: α_s·Ŝ
        if spatial_map is not None:
            spatial_flat = self._resize_and_flatten(spatial_map, h, w)
            guidance_factor = guidance_factor + self.alpha_spatial * spatial_flat
        
        # Relation guidance: α_r·max_k(R̂_k)
        if relation_map is not None:
            # Take max across relation channels (ignore channel 0 = NONE)
            if relation_map.shape[1] > 1:
                relation_max = relation_map[:, 1:, :, :].max(dim=1, keepdim=True)[0]
            else:
                relation_max = relation_map
            
            relation_flat = self._resize_and_flatten(relation_max, h, w)
            guidance_factor = guidance_factor + self.alpha_relation * relation_flat
        
        # Expand guidance to match coverage shape
        # cov: [B*nhead, T, H*W]
        # guidance_factor: [B, H*W]
        
        guidance_expanded = guidance_factor.unsqueeze(1).expand(-1, t, -1)  # [B, T, H*W]
        guidance_expanded = repeat(guidance_expanded, "b t l -> (b n) t l", n=self.nhead)
        
        # Apply guided penalty
        guided_cov = cov * guidance_expanded
        
        return guided_cov
    
    def _resize_and_flatten(self, feature_map: Tensor, h: int, w: int) -> Tensor:
        """Resize feature map to (h, w) and flatten to (B, H*W)."""
        if feature_map.dim() == 4:
            feature_map = feature_map.squeeze(1)  # [B, H', W']
        elif feature_map.dim() == 4 and feature_map.shape[1] == 1:
            feature_map = feature_map.squeeze(1)
        
        fm_h, fm_w = feature_map.shape[1], feature_map.shape[2]
        
        if fm_h != h or fm_w != w:
            feature_map = F.interpolate(
                feature_map.unsqueeze(1),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        return rearrange(feature_map, "b h w -> b (h w)")


def compute_guided_coverage_penalty(
    coverage: Tensor,
    spatial_map: Optional[Tensor] = None,
    relation_map: Optional[Tensor] = None,
    alpha_spatial: float = 0.3,
    alpha_relation: float = 0.2,
) -> Tensor:
    """
    Standalone function to compute guided coverage penalty.
    
    This can be used outside of ARM for debugging or alternative implementations.
    
    Args:
        coverage: Accumulated coverage [B, H*W]
        spatial_map: Spatial map [B, 1, H, W]
        relation_map: Relation map [B, C, H, W]
        alpha_spatial: Spatial guidance weight
        alpha_relation: Relation guidance weight
        
    Returns:
        Guided penalty [B, H*W]
    """
    B, L = coverage.shape
    guidance = torch.ones_like(coverage)
    
    if spatial_map is not None:
        S = spatial_map.view(B, -1)
        if S.shape[1] != L:
            # Need to resize
            H_s = int(spatial_map.shape[2])
            W_s = int(spatial_map.shape[3])
            H = int((L / W_s * H_s) ** 0.5)  # Approximate
            W = L // H
            S = F.interpolate(spatial_map, size=(H, W), mode='bilinear', align_corners=False)
            S = S.view(B, -1)
        
        guidance = guidance + alpha_spatial * S
    
    if relation_map is not None:
        # Max across relation channels (skip channel 0)
        if relation_map.shape[1] > 1:
            R_max = relation_map[:, 1:, :, :].max(dim=1, keepdim=True)[0]
        else:
            R_max = relation_map
        
        R = R_max.view(B, -1)
        if R.shape[1] != L:
            H_r = int(R_max.shape[2])
            W_r = int(R_max.shape[3])
            H = int((L / W_r * H_r) ** 0.5)
            W = L // H
            R = F.interpolate(R_max, size=(H, W), mode='bilinear', align_corners=False)
            R = R.view(B, -1)
        
        guidance = guidance + alpha_relation * R
    
    return coverage * guidance

