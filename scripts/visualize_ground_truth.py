#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from comer.datamodule.spatial_gt import AuxiliaryTargetGenerator


def extract_samples_from_zip(archive: ZipFile, folder: str, start_idx: int = 0, num_samples: int = 5):
    samples = []
    try:
        caption_path = f"data/{folder}/caption.txt"
        with archive.open(caption_path, 'r') as f:
            captions = f.readlines()
        
        count = 0
        for idx, line in enumerate(captions):
            if idx < start_idx:
                continue
            if count >= num_samples:
                break
            line = line.decode().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name = parts[0]
            formula = ' '.join(parts[1:])
            img_path = f"data/{folder}/img/{img_name}.bmp"
            try:
                with archive.open(img_path, 'r') as f:
                    img = Image.open(f).copy()
                    if img.mode != 'L':
                        img = img.convert('L')
                samples.append((idx, img_name, img, formula))
                count += 1
            except KeyError:
                continue
    except KeyError:
        print(f"Warning: Could not find {folder} in archive")
        return []
    return samples


def visualize_spatial_map(image_np: np.ndarray, spatial_map: np.ndarray, latex_str: str, output_path: str, sample_idx: int):
    if spatial_map.ndim == 3:
        spatial_map = spatial_map[0]
    
    img_normalized = image_np.astype(np.float32) / 255.0
    h_img, w_img = image_np.shape
    h_map, w_map = spatial_map.shape
    
    if h_map != h_img or w_map != w_img:
        spatial_upsampled = zoom(spatial_map, (h_img / h_map, w_img / w_map), order=1)
    else:
        spatial_upsampled = spatial_map
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    latex_display = latex_str[:60] + '...' if len(latex_str) > 60 else latex_str
    fig.suptitle(f'Sample {sample_idx}: Spatial Distribution Map\nLaTeX: {latex_display}', fontsize=14, fontweight='bold', y=0.98)
    
    axes[0].imshow(img_normalized, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Handwritten Image', fontsize=12, pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(img_normalized, cmap='gray', alpha=0.6, vmin=0, vmax=1)
    masked_spatial = np.ma.masked_where(spatial_upsampled < 0.05, spatial_upsampled)
    im = axes[1].imshow(masked_spatial, cmap='hot', alpha=0.7, vmin=0, vmax=max(spatial_map.max(), 0.1))
    coverage = ((spatial_map > 0.1).sum() / spatial_map.size) * 100
    axes[1].set_title(f'Spatial Map Overlay\nCoverage: {coverage:.1f}%', fontsize=12, pad=10)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize spatial maps for CoMER')
    parser.add_argument('--data_zip', type=str, default='data.zip', help='Path to data.zip')
    parser.add_argument('--output_dir', type=str, default='example', help='Output directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', '2014', '2016', '2019'])
    parser.add_argument('--start_idx', type=int, default=1000, help='Starting sample index')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--target_height', type=int, default=32)
    parser.add_argument('--target_width', type=int, default=64)
    args = parser.parse_args()
    
    data_zip_path = Path(args.data_zip)
    if not data_zip_path.exists():
        print(f"Error: data.zip not found at {data_zip_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SPATIAL MAP VISUALIZATION")
    print("=" * 80)
    print(f"Data zip: {data_zip_path}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.start_idx} to {args.start_idx + args.num_samples - 1}")
    print("=" * 80)
    
    with ZipFile(data_zip_path, 'r') as archive:
        samples = extract_samples_from_zip(archive, args.split, args.start_idx, args.num_samples)
        if not samples:
            print("No samples found!")
            sys.exit(1)
        
        print(f"\nVisualizing {len(samples)} samples...")
        
        for idx, img_name, img_pil, latex in samples:
            print(f"\nSample {idx}: {img_name}")
            print(f"LaTeX: {latex[:60]}{'...' if len(latex) > 60 else ''}")
            
            img_np = np.array(img_pil)
            img_w, img_h = img_pil.size
            
            generator = AuxiliaryTargetGenerator(
                img_height=img_h,
                img_width=img_w,
                target_height=args.target_height,
                target_width=args.target_width,
                device='cpu'
            )
            
            img_tensor = torch.from_numpy(img_np.astype(np.float32))
            S_target = generator(img_tensor, latex)
            spatial_map = S_target.numpy()
            
            print(f"Spatial map: shape={spatial_map.shape}, max={spatial_map.max():.3f}")
            
            output_path = output_dir / f'spatial_map_sample_{idx}.png'
            visualize_spatial_map(img_np, spatial_map, latex, str(output_path), idx)
            print(f"Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print(f"Output saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
