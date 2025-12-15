from typing import List, Tuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .spatial_head import SpatialPredictionHead


class CoMER(pl.LightningModule):
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
        use_spatial_aux: bool = False,
        spatial_hidden_channels: int = 256,
        use_spatial_guide: bool = False,
        spatial_scale: float = 1.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            use_spatial_guide=use_spatial_guide,
            spatial_scale=spatial_scale,
        )
        
        self.use_spatial_aux = use_spatial_aux
        self.use_spatial_guide = use_spatial_guide
        self.spatial_head = None
        if use_spatial_aux:
            self.spatial_head = SpatialPredictionHead(
                in_channels=d_model,
                hidden_channels=spatial_hidden_channels,
            )

    def forward(
        self, 
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        spatial_map_gt: Optional[FloatTensor] = None,
        return_spatial: bool = False,
    ) -> Tuple[FloatTensor, Optional[FloatTensor]]:
        feature, mask = self.encoder(img, img_mask)
        
        spatial_pred = None
        spatial_for_guide = None
        
        if self.use_spatial_aux and self.spatial_head is not None:
            spatial_pred = self.spatial_head(feature)
        
        if self.use_spatial_guide:
            if spatial_map_gt is not None:
                spatial_for_guide = spatial_map_gt
            elif spatial_pred is not None:
                spatial_for_guide = spatial_pred.detach()
        
        feature_doubled = torch.cat((feature, feature), dim=0)
        mask_doubled = torch.cat((mask, mask), dim=0)
        
        if spatial_for_guide is not None:
            spatial_doubled = torch.cat((spatial_for_guide, spatial_for_guide), dim=0)
        else:
            spatial_doubled = None
        
        out = self.decoder(feature_doubled, mask_doubled, tgt, spatial_map=spatial_doubled)
        
        if return_spatial:
            return out, spatial_pred
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
        **kwargs,
    ) -> List[Hypothesis]:
        feature, mask = self.encoder(img, img_mask)
        
        spatial_for_guide = None
        if self.use_spatial_guide:
            if spatial_map is not None:
                spatial_for_guide = spatial_map
            elif self.use_spatial_aux and self.spatial_head is not None:
                with torch.no_grad():
                    spatial_for_guide = self.spatial_head(feature)
        
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature,
            spatial_map=spatial_for_guide,
        )
    
    def predict_spatial(self, img: FloatTensor, img_mask: LongTensor) -> Optional[FloatTensor]:
        if not self.use_spatial_aux or self.spatial_head is None:
            return None
        feature, mask = self.encoder(img, img_mask)
        return self.spatial_head(feature)
