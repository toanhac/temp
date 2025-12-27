"""
Auxiliary Task Heads for Multi-Task Learning
=============================================

This module contains the prediction heads for:
1. SpatialHead: Predicts spatial distribution map [B, 1, H, W]
   - Uses CoordConv for position awareness
   - Uses DeformableConv for shape-adaptive receptive fields
2. RelationHead: Predicts multi-label relation map [B, 7, H, W]
   - Uses GlobalContextBlock for long-range dependencies

These heads are attached to the encoder output and trained with
auxiliary losses alongside the main recognition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torchvision.ops import DeformConv2d

from .multiscale_modules import CoordConv, GlobalContextBlock


class SpatialHead(nn.Module):
    """
    Spatial Head with CoordConv + DeformableConv.
    
    Improvements:
    - CoordConv: Adds (i,j) coordinate channels for absolute position awareness
    - DeformConv: Learns offset to adapt kernel to character shapes (curves, slants)
    
    Input: Encoder features [B, H, W, D] 
    Output: Spatial map [B, 1, H, W] with values in [0, 1]
    
    Loss: Smooth L1 Loss (robust to outliers)
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 64):
        super().__init__()
        
        # CoordConv: adds 2 coordinate channels, project to hidden_dim
        self.coord_conv = CoordConv(d_model, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Offset prediction for DeformConv
        # DeformConv2d with 3x3 kernel needs 2*3*3=18 offset channels
        self.offset_conv = nn.Conv2d(hidden_dim, 18, kernel_size=3, padding=1)
        
        # DeformableConv: shape-adaptive convolution
        self.deform_conv = DeformConv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Initialize offset to zero (start with regular conv behavior)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, H, W, D] from encoder (note: BHWD not BDHW)
            
        Returns:
            spatial_map: [B, 1, H, W] - where to focus (stroke regions)
        """
        # Permute from [B, H, W, D] to [B, D, H, W]
        x = features.permute(0, 3, 1, 2)
        
        # CoordConv layer
        x = F.relu(self.bn1(self.coord_conv(x)), inplace=True)
        
        # DeformableConv layer
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = F.relu(self.bn2(x), inplace=True)
        
        # Output
        return self.output(x)


class RelationHead(nn.Module):
    """
    Relation Head with Global Context Block.
    
    Improvements:
    - GlobalContextBlock: Lightweight global context modeling for long-range dependencies
      (e.g., numerator-denominator relationship in fractions)
    
    Each pixel can have MULTIPLE relation labels active simultaneously.
    For example, a pixel might be both INSIDE (sqrt) and ABOVE (numerator).
    
    Input: Encoder features [B, H, W, D]
    Output: Relation map [B, C, H, W] with values in [0, 1] for each class
    
    Classes (C=7):
        0: NONE (typically not used in loss)
        1: HORIZONTAL (baseline)
        2: ABOVE (numerator)
        3: BELOW (denominator)
        4: SUPERSCRIPT
        5: SUBSCRIPT
        6: INSIDE (sqrt, etc.)
    
    Loss: Binary Cross-Entropy (multi-label classification)
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_classes: int = 7, relation_hidden_dim: int = 64, use_hierarchical: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.relation_hidden_dim = relation_hidden_dim
        self.use_hierarchical = use_hierarchical
        
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.gc_block = GlobalContextBlock(hidden_dim, reduction_ratio=4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        if use_hierarchical:
            self.vertical_net = nn.Sequential(
                nn.Conv2d(2, relation_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(relation_hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.script_net = nn.Sequential(
                nn.Conv2d(2, relation_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(relation_hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.inside_net = nn.Sequential(
                nn.Conv2d(1, relation_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(relation_hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.alpha_vertical = nn.Parameter(torch.tensor(0.4))
            self.alpha_script = nn.Parameter(torch.tensor(0.3))
            self.alpha_inside = nn.Parameter(torch.tensor(0.3))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.gc_block(x)
        x = self.refine(x)
        return self.output(x)
    
    def compute_guidance(self, relation_map: torch.Tensor) -> torch.Tensor:
        if not self.use_hierarchical:
            return relation_map.max(dim=1, keepdim=True)[0] if relation_map.shape[1] > 1 else relation_map
        
        if relation_map.shape[1] < 7:
            return relation_map.max(dim=1, keepdim=True)[0] if relation_map.shape[1] > 1 else relation_map
        
        vertical = torch.cat([relation_map[:, 2:3], relation_map[:, 3:4]], dim=1)
        script = torch.cat([relation_map[:, 4:5], relation_map[:, 5:6]], dim=1)
        inside = relation_map[:, 6:7]
        
        g_v = self.vertical_net(vertical)
        g_s = self.script_net(script)
        g_i = self.inside_net(inside)
        
        alpha_sum = self.alpha_vertical.abs() + self.alpha_script.abs() + self.alpha_inside.abs()
        return (self.alpha_vertical.abs() * g_v + self.alpha_script.abs() * g_s + self.alpha_inside.abs() * g_i) / alpha_sum


def compute_spatial_loss(
    pred: torch.Tensor,   # [B, 1, H, W]
    target: torch.Tensor, # [B, 1, H, W]
    mask: torch.Tensor = None  # [B, H, W] valid pixels
) -> torch.Tensor:
    """
    Compute spatial recovery loss using Smooth L1.
    
    Smooth L1 is more robust to outliers than MSE.
    """
    if mask is not None:
        # Apply mask to ignore padding regions
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
        pred = pred * mask
        target = target * mask
        
        # Compute loss only on valid pixels
        loss = F.smooth_l1_loss(pred, target, reduction='sum')
        num_valid = mask.sum().clamp(min=1)
        return loss / num_valid
    else:
        return F.smooth_l1_loss(pred, target)


def compute_relation_loss(
    pred: torch.Tensor,   # [B, C, H, W]
    target: torch.Tensor, # [B, C, H, W]
    mask: torch.Tensor = None,  # [B, H, W]
    ignore_class_0: bool = True  # Ignore NONE class
) -> torch.Tensor:
    """
    Compute multi-label relation loss using Binary Cross-Entropy.
    
    Each channel is treated as an independent binary classification:
    - Channel k: Is this pixel part of relation k?
    
    This allows multiple channels to be active for the same pixel.
    """
    if ignore_class_0:
        # Skip channel 0 (NONE)
        pred = pred[:, 1:, :, :]
        target = target[:, 1:, :, :]
    
    if mask is not None:
        # Apply mask to ignore padding regions
        mask = mask.unsqueeze(1).expand_as(pred)  # [B, C, H, W]
        
        # Compute BCE only on valid pixels
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        bce = bce * mask
        loss = bce.sum() / mask.sum().clamp(min=1)
        return loss
    else:
        return F.binary_cross_entropy(pred, target)


class AuxiliaryHeads(nn.Module):
    """
    Combined auxiliary heads for multi-task learning.
    
    This module contains both SpatialHead and RelationHead,
    and provides methods for computing their losses.
    """
    
    def __init__(self, d_model: int = 256, num_relation_classes: int = 7):
        super().__init__()
        
        self.spatial_head = SpatialHead(d_model=d_model)
        self.relation_head = RelationHead(d_model=d_model, num_classes=num_relation_classes)
    
    def forward(
        self, 
        features: torch.Tensor  # [B, H, W, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Encoder output [B, H, W, D]
            
        Returns:
            spatial_pred: [B, 1, H, W]
            relation_pred: [B, C, H, W]
        """
        spatial_pred = self.spatial_head(features)
        relation_pred = self.relation_head(features)
        return spatial_pred, relation_pred
    
    def compute_loss(
        self,
        features: torch.Tensor,  # [B, H, W, D]
        spatial_gt: torch.Tensor,  # [B, 1, H', W']
        relation_gt: torch.Tensor,  # [B, C, H', W']
        mask: torch.Tensor = None,  # [B, H, W]
        lambda_s: float = 0.5,
        lambda_r: float = 0.3,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined auxiliary loss.
        
        Args:
            features: Encoder features
            spatial_gt: Ground truth spatial map
            relation_gt: Ground truth relation map
            mask: Valid pixel mask
            lambda_s: Weight for spatial loss
            lambda_r: Weight for relation loss
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual losses for logging
        """
        # Get predictions
        spatial_pred, relation_pred = self.forward(features)
        
        # Resize GT to match prediction size if needed
        pred_h, pred_w = spatial_pred.shape[-2:]
        gt_h, gt_w = spatial_gt.shape[-2:]
        
        if pred_h != gt_h or pred_w != gt_w:
            spatial_gt = F.interpolate(
                spatial_gt, size=(pred_h, pred_w), mode='bilinear', align_corners=False
            )
            relation_gt = F.interpolate(
                relation_gt, size=(pred_h, pred_w), mode='bilinear', align_corners=False
            )
        
        # Compute losses
        loss_spatial = compute_spatial_loss(spatial_pred, spatial_gt, mask)
        loss_relation = compute_relation_loss(relation_pred, relation_gt, mask, ignore_class_0=True)
        
        # Combined loss
        total_loss = lambda_s * loss_spatial + lambda_r * loss_relation
        
        loss_dict = {
            'loss_spatial': loss_spatial.detach(),
            'loss_relation': loss_relation.detach(),
            'loss_aux': total_loss.detach(),
        }
        
        return total_loss, loss_dict


def get_max_relation_map(relation_pred: torch.Tensor) -> torch.Tensor:
    """
    Get maximum relation probability across all channels.
    
    This is used in Guided Coverage Attention:
        max_k R̂_{k,i} = maximum probability at position i
    
    Args:
        relation_pred: [B, C, H, W] multi-label predictions
        
    Returns:
        max_relation: [B, H*W] flattened max values
    """
    # Take max across channel dimension, ignoring channel 0 (NONE)
    max_relation = relation_pred[:, 1:, :, :].max(dim=1)[0]  # [B, H, W]
    
    # Flatten for attention computation
    B, H, W = max_relation.shape
    return max_relation.view(B, H * W)  # [B, H*W]


if __name__ == '__main__':
    # Test the heads
    batch_size = 2
    H, W, D = 4, 8, 256
    
    features = torch.randn(batch_size, H, W, D)
    
    print("Testing AuxiliaryHeads")
    print("=" * 50)
    print(f"Input features: {features.shape}")
    
    heads = AuxiliaryHeads(d_model=D)
    spatial_pred, relation_pred = heads(features)
    
    print(f"\nSpatial prediction: {spatial_pred.shape}")
    print(f"  Range: [{spatial_pred.min():.4f}, {spatial_pred.max():.4f}]")
    
    print(f"\nRelation prediction: {relation_pred.shape}")
    print(f"  Range: [{relation_pred.min():.4f}, {relation_pred.max():.4f}]")
    
    # Test loss computation
    spatial_gt = torch.rand(batch_size, 1, H, W)
    relation_gt = torch.rand(batch_size, 7, H, W)
    
    total_loss, loss_dict = heads.compute_loss(
        features, spatial_gt, relation_gt
    )
    
    print(f"\nLoss computation:")
    print(f"  Spatial loss: {loss_dict['loss_spatial']:.4f}")
    print(f"  Relation loss: {loss_dict['loss_relation']:.4f}")
    print(f"  Total aux loss: {loss_dict['loss_aux']:.4f}")
    
    # Test max relation for guided coverage
    max_rel = get_max_relation_map(relation_pred)
    print(f"\nMax relation map: {max_rel.shape}")
    print(f"  For guided coverage attention")
