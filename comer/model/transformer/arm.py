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
        use_spatial_guide: bool = False,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        self.use_spatial_guide = use_spatial_guide
        self.spatial_scale = spatial_scale

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
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        
        if self.use_spatial_guide and spatial_map is not None:
            cov = self._apply_spatial_guidance(cov, spatial_map, b, t, h, w)
        
        return cov
    
    def _apply_spatial_guidance(
        self, 
        cov: Tensor, 
        spatial_map: Tensor, 
        b: int, 
        t: int, 
        h: int, 
        w: int
    ) -> Tensor:
        if spatial_map.dim() == 4:
            spatial_map = spatial_map.squeeze(1)
        
        sm_h, sm_w = spatial_map.shape[1], spatial_map.shape[2]
        if sm_h != h or sm_w != w:
            spatial_map = F.interpolate(
                spatial_map.unsqueeze(1),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        spatial_flat = rearrange(spatial_map, "b h w -> b (h w)")
        spatial_expanded = spatial_flat.unsqueeze(1).expand(-1, t, -1)
        spatial_expanded = repeat(spatial_expanded, "b t l -> (b n) t l", n=self.nhead)
        
        scaled_cov = cov * (1.0 + self.spatial_scale * spatial_expanded)
        
        return scaled_cov
