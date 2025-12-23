from typing import List, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .auxiliary_heads import SpatialHead, RelationHead


class CoMER(pl.LightningModule):
    """
    CoMER with Multi-Task Learning and Guided Coverage.
    
    Features:
    - 16x→8x Feature Fusion for higher resolution
    - Spatial prediction with CoordConv + DeformableConv
    - Relation prediction with GlobalContextBlock
    - Guided coverage attention
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
        # Feature fusion
        fusion_out_channels: int = 128,
        # Auxiliary task options
        use_spatial_aux: bool = False,
        use_relation_aux: bool = False,
        spatial_hidden_channels: int = 64,  # Reduced for bottleneck
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
            d_model=d_model, 
            growth_rate=growth_rate, 
            num_layers=num_layers,
            fusion_out_channels=fusion_out_channels,
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
        
        # Auxiliary heads (using enhanced versions from auxiliary_heads.py)
        self.spatial_head = None
        self.relation_head = None
        
        if use_spatial_aux:
            self.spatial_head = SpatialHead(
                d_model=d_model,
                hidden_dim=spatial_hidden_channels,
            )
        
        if use_relation_aux:
            self.relation_head = RelationHead(
                d_model=d_model,
                hidden_dim=relation_hidden_channels,
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

