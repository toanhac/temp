import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import FloatTensor


class SpatialPredictionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.conv3 = nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels // 4)
        self.output = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)
        
    def forward(self, encoder_features: FloatTensor) -> FloatTensor:
        if encoder_features.dim() == 4:
            b, h, w, d = encoder_features.shape
            x = rearrange(encoder_features, 'b h w d -> b d h w')
        else:
            x = encoder_features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.output(x))
        return x


class SpatialAwareCoverageModule(nn.Module):
    def __init__(self, use_spatial_scaling: bool = True, coverage_lambda: float = 1.0):
        super().__init__()
        self.use_spatial_scaling = use_spatial_scaling
        self.coverage_lambda = coverage_lambda
        
    def forward(self, coverage: FloatTensor, spatial_map: FloatTensor = None) -> FloatTensor:
        if not self.use_spatial_scaling or spatial_map is None:
            return coverage * self.coverage_lambda
        if spatial_map.dim() == 4:
            spatial_map = spatial_map.squeeze(1)
        b, h, w = spatial_map.shape
        spatial_flat = rearrange(spatial_map, 'b h w -> b (h w)')
        if coverage.dim() == 3 and coverage.shape[-1] == h * w:
            penalty = spatial_flat.unsqueeze(1) * coverage
        else:
            penalty = coverage
        return penalty * self.coverage_lambda
