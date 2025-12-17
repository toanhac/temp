"""
Auxiliary Task Heads for Multi-Task Learning
=============================================

This module contains the prediction heads for:
1. SpatialHead: Predicts spatial distribution map [B, 1, H, W]
2. RelationHead: Predicts multi-label relation map [B, 7, H, W]

These heads are attached to the encoder output and trained with
auxiliary losses alongside the main recognition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialHead(nn.Module):
    """
    Predicts spatial distribution (stroke density) map from encoder features.
    
    This head learns to distinguish foreground (strokes) from background,
    forcing the encoder to learn localized features.
    
    Input: Encoder features [B, H, W, D] 
    Output: Spatial map [B, 1, H, W] with values in [0, 1]
    
    Loss: Smooth L1 Loss (robust to outliers)
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 64):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Feature compression
            nn.Conv2d(d_model, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Refinement
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Output
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
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, H, W, D] from encoder (note: BHWD not BDHW)
            
        Returns:
            spatial_map: [B, 1, H, W] - where to focus (stroke regions)
        """
        # Permute from [B, H, W, D] to [B, D, H, W]
        x = features.permute(0, 3, 1, 2)
        return self.conv(x)


class RelationHead(nn.Module):
    """
    Predicts multi-label relation map from encoder features.
    
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
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_classes: int = 7):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.conv = nn.Sequential(
            # Feature compression
            nn.Conv2d(d_model, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Refinement with larger receptive field
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Multi-label output
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            nn.Sigmoid()  # Multi-label, NOT softmax
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
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, H, W, D] from encoder
            
        Returns:
            relation_map: [B, C, H, W] - multi-label relation probabilities
        """
        x = features.permute(0, 3, 1, 2)  # [B, D, H, W]
        return self.conv(x)


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
