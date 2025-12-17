from typing import List, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .spatial_head import SpatialPredictionHead


class RelationPredictionHead(nn.Module):
    """
    Predicts multi-label relation map from encoder features.
    
    Outputs [B, num_classes, H, W] with independent sigmoid activations
    for multi-label classification.
    """
    
    def __init__(
        self, 
        in_channels: int = 256, 
        hidden_channels: int = 128,
        num_classes: int = 7,
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
            nn.Sigmoid(),  # Multi-label (independent per class)
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
    
    def forward(self, feature: FloatTensor) -> FloatTensor:
        """
        Args:
            feature: [B, H, W, D] encoder output
            
        Returns:
            relation_map: [B, num_classes, H, W]
        """
        # Permute from [B, H, W, D] to [B, D, H, W]
        x = feature.permute(0, 3, 1, 2)
        return self.conv(x)


class CoMER(pl.LightningModule):
    """
    CoMER with Multi-Task Learning and Guided Coverage.
    
    Supports:
    - Spatial prediction auxiliary task
    - Relation prediction auxiliary task  
    - Guided coverage attention using spatial and relation maps
    """
    
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # Auxiliary task options
        use_spatial_aux: bool = False,
        use_relation_aux: bool = False,
        spatial_hidden_channels: int = 256,
        relation_hidden_channels: int = 128,
        num_relation_classes: int = 7,
        # Guided coverage options
        use_spatial_guide: bool = False,
        use_guided_coverage: bool = False,
        spatial_scale: float = 1.0,
        alpha_spatial: float = 0.3,
        alpha_relation: float = 0.2,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        
        # Determine decoder guidance mode
        decoder_use_spatial_guide = use_spatial_guide or use_guided_coverage
        
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            use_spatial_guide=decoder_use_spatial_guide,
            spatial_scale=spatial_scale,
            # Pass guided coverage params if supported by decoder
            use_guided_coverage=use_guided_coverage,
            alpha_spatial=alpha_spatial,
            alpha_relation=alpha_relation,
        )
        
        # Config flags
        self.use_spatial_aux = use_spatial_aux
        self.use_relation_aux = use_relation_aux
        self.use_spatial_guide = use_spatial_guide
        self.use_guided_coverage = use_guided_coverage
        
        # Auxiliary heads
        self.spatial_head = None
        self.relation_head = None
        
        if use_spatial_aux:
            self.spatial_head = SpatialPredictionHead(
                in_channels=d_model,
                hidden_channels=spatial_hidden_channels,
            )
        
        if use_relation_aux:
            self.relation_head = RelationPredictionHead(
                in_channels=d_model,
                hidden_channels=relation_hidden_channels,
                num_classes=num_relation_classes,
            )

    def forward(
        self, 
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        spatial_map_gt: Optional[FloatTensor] = None,
        relation_map_gt: Optional[FloatTensor] = None,
        return_spatial: bool = False,
        return_relation: bool = False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, ...]]:
        """
        Forward pass with optional auxiliary outputs.
        
        Returns:
            If return_spatial and return_relation:
                (decoder_output, spatial_pred, relation_pred)
            If only return_spatial:
                (decoder_output, spatial_pred)
            If only return_relation:
                (decoder_output, relation_pred)
            Otherwise:
                decoder_output
        """
        feature, mask = self.encoder(img, img_mask)
        
        # Compute auxiliary predictions
        spatial_pred = None
        relation_pred = None
        
        if self.use_spatial_aux and self.spatial_head is not None:
            spatial_pred = self.spatial_head(feature)
        
        if self.use_relation_aux and self.relation_head is not None:
            relation_pred = self.relation_head(feature)
        
        # Determine guidance maps for decoder
        spatial_for_guide = None
        relation_for_guide = None
        
        if self.use_spatial_guide or self.use_guided_coverage:
            # Prefer GT during training, fallback to predictions
            if spatial_map_gt is not None:
                spatial_for_guide = spatial_map_gt
            elif spatial_pred is not None:
                spatial_for_guide = spatial_pred.detach()
            
            if self.use_guided_coverage:
                if relation_map_gt is not None:
                    relation_for_guide = relation_map_gt
                elif relation_pred is not None:
                    relation_for_guide = relation_pred.detach()
        
        # Double features for bi-directional decoding
        feature_doubled = torch.cat((feature, feature), dim=0)
        mask_doubled = torch.cat((mask, mask), dim=0)
        
        if spatial_for_guide is not None:
            spatial_doubled = torch.cat((spatial_for_guide, spatial_for_guide), dim=0)
        else:
            spatial_doubled = None
        
        if relation_for_guide is not None:
            relation_doubled = torch.cat((relation_for_guide, relation_for_guide), dim=0)
        else:
            relation_doubled = None
        
        # Decode
        out = self.decoder(
            feature_doubled, mask_doubled, tgt, 
            spatial_map=spatial_doubled,
            relation_map=relation_doubled,
        )
        
        # Return requested outputs
        if return_spatial and return_relation:
            return out, spatial_pred, relation_pred
        elif return_spatial:
            return out, spatial_pred
        elif return_relation:
            return out, relation_pred
        return out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        spatial_map: Optional[FloatTensor] = None,
        relation_map: Optional[FloatTensor] = None,
        **kwargs,
    ) -> List[Hypothesis]:
        """Beam search with optional guided coverage."""
        feature, mask = self.encoder(img, img_mask)
        
        # Determine guidance maps
        spatial_for_guide = None
        relation_for_guide = None
        
        if self.use_spatial_guide or self.use_guided_coverage:
            if spatial_map is not None:
                spatial_for_guide = spatial_map
            elif self.use_spatial_aux and self.spatial_head is not None:
                with torch.no_grad():
                    spatial_for_guide = self.spatial_head(feature)
            
            if self.use_guided_coverage:
                if relation_map is not None:
                    relation_for_guide = relation_map
                elif self.use_relation_aux and self.relation_head is not None:
                    with torch.no_grad():
                        relation_for_guide = self.relation_head(feature)
        
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature,
            spatial_map=spatial_for_guide,
            relation_map=relation_for_guide,
        )
    
    def predict_spatial(self, img: FloatTensor, img_mask: LongTensor) -> Optional[FloatTensor]:
        """Predict spatial map for a given image."""
        if not self.use_spatial_aux or self.spatial_head is None:
            return None
        feature, mask = self.encoder(img, img_mask)
        return self.spatial_head(feature)
    
    def predict_relation(self, img: FloatTensor, img_mask: LongTensor) -> Optional[FloatTensor]:
        """Predict relation map for a given image."""
        if not self.use_relation_aux or self.relation_head is None:
            return None
        feature, mask = self.encoder(img, img_mask)
        return self.relation_head(feature)

