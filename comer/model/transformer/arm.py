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
    def __init__(
        self, 
        nhead: int, 
        dc: int, 
        cross_coverage: bool, 
        self_coverage: bool,
        use_guided_coverage: bool = False,
        alpha_spatial: float = 0.3,
        alpha_relation: float = 0.2,
        dynamic_weighting: bool = True,
        decay_tau_ratio: float = 3.0,
        coverage_aware_w1: float = 2.0,
        coverage_aware_w2: float = 1.0,
        use_spatial_guide: bool = False,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        
        self.use_guided_coverage = use_guided_coverage or use_spatial_guide
        self.alpha_spatial = alpha_spatial if use_guided_coverage else spatial_scale
        self.alpha_relation = alpha_relation
        self.dynamic_weighting = dynamic_weighting
        self.decay_tau_ratio = decay_tau_ratio
        self.coverage_aware_w1 = coverage_aware_w1
        self.coverage_aware_w2 = coverage_aware_w2

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
        epoch_idx: int = -1,
    ) -> Tensor:
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

        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        cov = self.conv(attns)
        cov = self.act(cov)
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)
        
        coverage_2d = cov
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        
        if self.use_guided_coverage and (spatial_map is not None or relation_map is not None):
            guidance = self._compute_guidance(
                spatial_map, relation_map, coverage_2d, b, t, h, w, epoch_idx
            )
            cov = cov + guidance
        
        return cov
    
    def _compute_guidance(
        self, 
        spatial_map: Optional[Tensor],
        relation_map: Optional[Tensor],
        coverage_2d: Tensor,
        b: int, 
        t: int, 
        h: int, 
        w: int,
        epoch_idx: int = -1,
    ) -> Tensor:
        guidance = torch.zeros(b * self.nhead, t, h * w, device=coverage_2d.device, dtype=coverage_2d.dtype)
        
        beta = self._compute_dynamic_weight(epoch_idx) if self.dynamic_weighting else 1.0
        
        if spatial_map is not None:
            spatial_resized = self._resize_map(spatial_map, h, w)
            coverage_normalized = coverage_2d.mean(dim=1, keepdim=True)
            coverage_resized = F.interpolate(coverage_normalized, size=(h, w), mode='bilinear', align_corners=False)
            coverage_resized = rearrange(coverage_resized, "(b t) c h w -> b t c h w", b=b)
            
            spatial_expanded = spatial_resized.unsqueeze(1).expand(-1, t, -1, -1, -1)
            
            G_s = self.coverage_aware_w1 * spatial_expanded * (1 - coverage_resized) - \
                  self.coverage_aware_w2 * (1 - spatial_expanded) * coverage_resized
            
            G_s_flat = rearrange(G_s, "b t c h w -> b t (h w c)")
            G_s_flat = repeat(G_s_flat, "b t l -> (b n) t l", n=self.nhead)
            guidance = guidance + self.alpha_spatial * G_s_flat
        
        if relation_map is not None:
            R_guidance = self._compute_relation_guidance(relation_map, h, w, b, t)
            guidance = guidance + self.alpha_relation * R_guidance
        
        return beta * guidance
    
    def _compute_dynamic_weight(self, epoch_idx: int, max_epochs: int = 300) -> float:
        if epoch_idx < 0:
            return 0.0
        tau = max_epochs / self.decay_tau_ratio
        return torch.exp(torch.tensor(-epoch_idx / tau, dtype=torch.float32)).item()
    
    def _compute_relation_guidance(self, relation_map: Tensor, h: int, w: int, b: int, t: int) -> Tensor:
        if relation_map.shape[1] > 1:
            relation_max = relation_map[:, 1:, :, :].max(dim=1, keepdim=True)[0]
        else:
            relation_max = relation_map
        
        relation_resized = self._resize_map(relation_max, h, w)
        relation_flat = rearrange(relation_resized, "b c h w -> b (h w c)")
        relation_flat = relation_flat.unsqueeze(1).expand(-1, t, -1)
        relation_flat = repeat(relation_flat, "b t l -> (b n) t l", n=self.nhead)
        return relation_flat
    
    def _resize_map(self, feature_map: Tensor, h: int, w: int) -> Tensor:
        if feature_map.shape[2:] != (h, w):
            feature_map = F.interpolate(feature_map, size=(h, w), mode='bilinear', align_corners=False)
        return feature_map

