# CoMER-SA: Coverage-Based Transformer with Spatial-Aware Auxiliary Task for Handwritten Mathematical Expression Recognition

---

## Abstract

We propose **CoMER-SA** (CoMER with Spatial-Aware Auxiliary Task), an enhanced architecture for Handwritten Mathematical Expression Recognition (HMER) that integrates a spatial prediction auxiliary task into the CoMER framework. Our approach leverages multi-task learning to explicitly guide the model's attention toward symbol regions while maintaining the coverage mechanism's ability to prevent attention drift. By predicting symbol spatial distribution maps as an auxiliary task, CoMER-SA learns richer structural representations that improve recognition accuracy, particularly for expressions with complex spatial arrangements such as fractions, superscripts, and subscripts.

---

## 1. Introduction

### 1.1 Problem Statement

Handwritten Mathematical Expression Recognition (HMER) presents unique challenges due to the complex two-dimensional spatial relationships between symbols. Unlike standard text recognition, mathematical expressions contain:

- **Nested structures**: Fractions within fractions, nested radicals
- **Spatial relationships**: Superscripts, subscripts, and operator placement
- **Variable symbol sizes**: Different scales for main expressions and sub-expressions

Standard attention-based encoder-decoder models often suffer from **attention drift**, where the attention mechanism repeatedly focuses on the same regions while missing other important symbols.

### 1.2 Motivation

CoMER (Coverage-based Transformer) addresses attention drift through a coverage mechanism that tracks cumulative attention and penalizes over-attended regions. However, this mechanism treats all spatial positions equally, without considering the actual importance of each position.

We hypothesize that explicitly modeling **where symbols are located** (symbol spatial distribution) can:
1. Provide additional structural supervision during training
2. Guide the coverage mechanism to prioritize stroke regions
3. Improve recognition of spatially complex expressions

---

## 2. Architecture

### 2.1 Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       CoMER-SA Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Input Image [B, 1, H, W]                                 │
│           │                                                 │
│           ▼                                                 │
│    ┌─────────────────┐                                      │
│    │  DenseNet-B     │                                      │
│    │  Encoder        │                                      │
│    └────────┬────────┘                                      │
│             │                                               │
│             ▼                                               │
│    ┌─────────────────┐     ┌─────────────────────────┐     │
│    │ Encoder Feature │────►│ Spatial Prediction Head  │     │
│    │ [B, H/8, W/8, D]│     │ (Auxiliary Task)         │     │
│    └────────┬────────┘     └───────────┬─────────────┘     │
│             │                          │                    │
│             │                          ▼                    │
│             │              ┌─────────────────────────┐     │
│             │              │ S_pred [B, 1, H/8, W/8] │     │
│             │              └───────────┬─────────────┘     │
│             │                          │                    │
│             ▼                          │ L_spatial          │
│    ┌─────────────────┐                 │                    │
│    │ Transformer     │                 │                    │
│    │ Decoder + ARM   │                 │                    │
│    └────────┬────────┘                 │                    │
│             │                          │                    │
│             ▼                          │                    │
│    ┌─────────────────┐                 │                    │
│    │ Logits          │                 │                    │
│    │ [B, L, V]       │                 │                    │
│    └────────┬────────┘                 │                    │
│             │                          │                    │
│             │ L_rec (CE Loss)          │                    │
│             │                          │                    │
│             └──────────┬───────────────┘                    │
│                        ▼                                    │
│             ┌─────────────────────────┐                     │
│             │ Total Loss              │                     │
│             │ L = L_rec + λ * L_spat  │                     │
│             └─────────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Components

#### 2.2.1 Encoder (DenseNet-B)

The encoder is based on DenseNet-B architecture with the following specifications:

| Parameter | Value |
|-----------|-------|
| Growth rate | 24 |
| Number of layers | 16 |
| Initial channels | 48 |
| Output channels | 684 |
| Downsampling factor | 8x |

The encoder produces feature maps F ∈ ℝ^(B×H/8×W/8×D) where D = 256 after projection.

#### 2.2.2 Spatial Prediction Head

The Spatial Prediction Head is a lightweight convolutional network that predicts symbol spatial distribution maps:

```
Input: F ∈ ℝ^(B×H×W×D)
       │
       ▼ (reshape to B×D×H×W)
┌──────────────────────────────────┐
│ Conv2D(D → 256, 3×3) + BN + ReLU │
├──────────────────────────────────┤
│ Conv2D(256 → 128, 3×3) + BN + ReLU│
├──────────────────────────────────┤
│ Conv2D(128 → 64, 3×3) + BN + ReLU │
├──────────────────────────────────┤
│ Conv2D(64 → 1, 1×1) + Sigmoid    │
└──────────────────────────────────┘
       │
       ▼
Output: S_pred ∈ [0, 1]^(B×1×H×W)
```

**Design rationale:**
- Multi-stage convolutions capture spatial patterns at different scales
- BatchNorm ensures training stability
- Sigmoid activation constrains output to [0, 1] probability range

#### 2.2.3 Transformer Decoder with ARM

The decoder consists of stacked transformer layers with the Attention Refinement Module (ARM):

- **Self-Attention**: Standard causal self-attention for autoregressive decoding
- **Cross-Attention**: Attends to encoder features with coverage mechanism
- **ARM**: Refines attention weights based on cumulative attention
- **Spatial-Guided Coverage**: Scales coverage penalty by spatial importance

#### 2.2.4 Spatial-Guided Coverage (NEW)

When `use_spatial_guide=True`, the ARM module uses spatial maps to modulate coverage penalty:

```
Standard Coverage (CoMER baseline):
    penalty = f(cumulative_attention)
    
Spatial-Guided Coverage:
    penalty = f(cumulative_attention) × (1 + α × spatial_map)
    
    where:
        spatial_map[i,j] = 1.0 for stroke regions
        spatial_map[i,j] = 0.0 for empty regions
        α = spatial_scale (default: 1.0)
```

**Effect:**
- **Stroke regions** (spatial=1): penalty scaled by `(1 + α)` → **stronger penalty** → avoid re-attending
- **Empty regions** (spatial=0): penalty scaled by `1` → normal penalty

```python
# In ARM.forward()
def _apply_spatial_guidance(self, cov, spatial_map, ...):
    spatial_expanded = expand_spatial_to_attention_shape(spatial_map)
    scaled_cov = cov * (1.0 + self.spatial_scale * spatial_expanded)
    return scaled_cov
```

**Benefits:**
1. Decoder knows "where strokes are" → avoids repeating attention on same symbol
2. Multi-stroke symbols (fractions, integrals) handled better
3. Complements multi-task learning - spatial info used in BOTH encoder AND decoder

#### 2.2.5 Loss Function

The total loss combines recognition loss and spatial prediction loss:

```
L_total = L_rec + λ * L_spatial

where:
    L_rec = CrossEntropy(logits, target_tokens)
    L_spatial = SmoothL1(S_pred, S_gt)
    λ = 0.5 (default)
```

**Smooth L1 Loss** is chosen for spatial prediction because:
1. More robust to outliers than MSE
2. Gradients don't explode for large differences
3. Better handles binary-like targets

---

## 3. Symbol Spatial Distribution Map

### 3.1 Ground Truth Generation

The spatial distribution map S_gt represents the location of handwritten strokes:

```
S_gt[i, j] = 1.0  if position (i, j) contains handwritten stroke
S_gt[i, j] = 0.0  otherwise
```

**Generation Process:**

1. **Binarization**: Apply Sauvola thresholding (or adaptive thresholding)
2. **Connected Component Analysis**: Extract individual stroke components
3. **Noise Removal**: Filter components smaller than minimum size (5 pixels)
4. **Downsampling**: Resize to target resolution matching encoder output

```python
class SpatialMapGenerator:
    def __init__(self, img_height, img_width, target_height, target_width):
        self.target_height = target_height  # H/8
        self.target_width = target_width    # W/8
    
    def __call__(self, image):
        binary = threshold(image)  # Sauvola thresholding
        components = connected_components(binary)
        mask = filter_small_components(components)
        spatial_map = downsample(mask, self.target_height, self.target_width)
        return spatial_map  # [1, H/8, W/8]
```

### 3.2 Properties

| Property | Value | Description |
|----------|-------|-------------|
| Shape | [1, H/8, W/8] | Matches encoder output resolution |
| Value range | [0, 1] | Binary after thresholding |
| Coverage | 5-25% | Typical percentage of active pixels |

---

## 4. Training

### 4.1 Multi-Task Learning Framework

```
Training batch:
    images:      [B, 1, H, W]
    tokens:      [B, L]
    spatial_gt:  [B, 1, H/8, W/8]

Forward pass:
    feature, mask = encoder(images)
    spatial_pred = spatial_head(feature)
    logits = decoder(feature, mask, tokens)

Loss computation:
    L_rec = CrossEntropy(logits, tokens)
    L_spatial = SmoothL1(spatial_pred, spatial_gt)
    L_total = L_rec + λ * L_spatial

Backward pass:
    ∂L_total/∂θ → Update all parameters
```

### 4.2 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| use_spatial_aux | True | Enable spatial auxiliary task (multi-task) |
| spatial_hidden_channels | 256 | Hidden channels in spatial head |
| spatial_loss_weight (λ) | 0.5 | Weight for spatial loss |
| use_spatial_guide | True | Enable spatial-guided coverage (guided decoding) |
| spatial_scale (α) | 1.0 | Scaling factor for spatial guidance |
| learning_rate | 0.08 | SGD learning rate |
| batch_size | 8 | Training batch size |

### 4.3 Configuration

```yaml
model:
  d_model: 256
  growth_rate: 24
  num_layers: 16
  nhead: 8
  num_decoder_layers: 3
  dim_feedforward: 1024
  dropout: 0.3
  dc: 32
  cross_coverage: true
  self_coverage: true
  # Multi-task Learning (encoder)
  use_spatial_aux: true
  spatial_hidden_channels: 256
  spatial_loss_weight: 0.5
  # Guided Decoding (decoder)
  use_spatial_guide: true
  spatial_scale: 1.0
  
data:
  use_spatial_maps: true
  spatial_cache_dir: data/cached_maps
```

---

## 5. Implementation

### 5.1 File Structure

```
comer/
├── model/
│   ├── comer.py           # Main model (modified)
│   ├── encoder.py         # DenseNet encoder
│   ├── decoder.py         # Transformer decoder
│   ├── spatial_head.py    # NEW: Spatial prediction head
│   └── transformer/
│       ├── arm.py         # Attention Refinement Module
│       └── attention.py   # Multi-head attention
├── datamodule/
│   ├── datamodule.py      # Data loading (modified)
│   ├── spatial_gt.py      # Spatial map generation
│   └── vocab.py           # Vocabulary
└── lit_comer.py           # Lightning module (modified)
```

### 5.2 Key Code Components

**SpatialPredictionHead** (`comer/model/spatial_head.py`):
```python
class SpatialPredictionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels//2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels//2, hidden_channels//4, 3, padding=1)
        self.output = nn.Conv2d(hidden_channels//4, 1, 1)
    
    def forward(self, encoder_features):
        x = rearrange(encoder_features, 'b h w d -> b d h w')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.sigmoid(self.output(x))
```

**Modified CoMER** (`comer/model/comer.py`):
```python
class CoMER(pl.LightningModule):
    def __init__(self, ..., use_spatial_aux=False):
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        if use_spatial_aux:
            self.spatial_head = SpatialPredictionHead(d_model, 256)
    
    def forward(self, img, img_mask, tgt, return_spatial=False):
        feature, mask = self.encoder(img, img_mask)
        spatial_pred = self.spatial_head(feature) if self.use_spatial_aux else None
        logits = self.decoder(feature, mask, tgt)
        if return_spatial:
            return logits, spatial_pred
        return logits
```

---

## 6. Theoretical Analysis

### 6.1 Multi-Task Learning Benefits

The spatial prediction task provides:

1. **Regularization**: Additional supervision prevents overfitting
2. **Feature enrichment**: Encoder learns to produce features useful for both tasks
3. **Implicit attention guidance**: Spatial awareness improves attention quality

### 6.2 Gradient Flow

```
∂L_total/∂θ_encoder = ∂L_rec/∂θ_encoder + λ * ∂L_spatial/∂θ_encoder
                       ↑                      ↑
                  recognition signal    spatial signal
```

Both tasks contribute gradients to the encoder, encouraging:
- Feature discrimination for symbol classification (from L_rec)
- Spatial localization capability (from L_spatial)

### 6.3 Expected Improvements

| Expression Type | Expected Gain | Rationale |
|-----------------|---------------|-----------|
| Fractions | +0.5-1.0% | Multi-level structures benefit from spatial awareness |
| Superscripts/Subscripts | +0.5-1.0% | Explicit position modeling |
| Nested expressions | +0.3-0.8% | Complex spatial patterns |
| Simple expressions | +0.1-0.3% | Already well-handled by baseline |

---

## 7. Usage

### 7.1 Training

```bash
# Standard training with spatial auxiliary
python train.py --config config.yaml

# With custom spatial loss weight
python train.py --config config.yaml \
    --model.use_spatial_aux true \
    --model.spatial_loss_weight 0.5
```

### 7.2 Pre-generate Spatial Maps (Optional)

```bash
# Generate cached spatial maps for faster training
python scripts/pregenerate_ground_truth.py \
    --data_zip data.zip \
    --output_dir data/cached_maps \
    --workers 8
```

### 7.3 Visualization

```bash
# Visualize spatial maps
python scripts/visualize_ground_truth.py \
    --data_zip data.zip \
    --split train \
    --num_samples 10
```

---

## 8. Conclusion

CoMER-SA extends the CoMER architecture with a spatial-aware auxiliary task that:

1. **Adds explicit spatial supervision** through symbol distribution map prediction
2. **Uses multi-task learning** to enhance encoder representations
3. **Maintains computational efficiency** with a lightweight prediction head
4. **Improves recognition** of spatially complex expressions

The approach is complementary to the existing coverage mechanism and can be easily integrated into any encoder-decoder HMER model.

---

## References

1. Zhao, W., & Gao, L. (2022). CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition. ECCV.

2. Zhang, J., Du, J., & Dai, L. (2018). Multi-Scale Attention with Dense Encoder for Handwritten Mathematical Expression Recognition. ICPR.

3. Li, H., et al. (2022). Counting-Aware Network for Handwritten Mathematical Expression Recognition. ECCV.

4. Yuan, Y., et al. (2022). Syntax-Aware Network for Handwritten Mathematical Expression Recognition. CVPR.

5. SSAN Paper (2025). Symbol Spatial-Aware Network for Handwritten Mathematical Expression Recognition. AAAI.
