#!/usr/bin/env python3
"""
Pre-generate Ground Truth Maps for CoMER Multi-Task Training
=============================================================

This script pre-generates:
1. Spatial maps (S): Binary masks showing symbol regions
2. Relation maps (R): MULTI-LABEL multi-channel masks showing structural relations

Size modes:
- 'fixed': Use fixed target size (default: 32x64) - higher resolution
- 'dynamic': Use dynamic size based on encoder downsample factor (16x)

Ground truth types:
- 'spatial': Only generate spatial distribution maps
- 'relation': Only generate relation structure maps  
- 'both': Generate both types (default)

Usage:
    # Generate both types with dynamic size (matches encoder output EXACTLY)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --size_mode dynamic
    
    # Generate with fixed size (for visualization)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --size_mode fixed
    
    # Generate only relation maps with multi-label
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --type relation
"""

import os
import sys
import argparse
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from comer.datamodule.spatial_gt import SpatialMapGenerator
from comer.datamodule.multilabel_relation_gt import (
    MultiLabelRelationMapGenerator as RelationMapGenerator,
    RelationType
)
USING_MULTILABEL = True


# Encoder downsampling analysis:
# conv1(stride=2) -> maxpool(2) -> trans1(avgpool2) -> trans2(avgpool2)
# Total: 2 * 2 * 2 * 2 = 16x downsampling with ceil_mode
ENCODER_DOWNSAMPLE_FACTOR = 16
DEFAULT_FIXED_HEIGHT = 32
DEFAULT_FIXED_WIDTH = 64


def compute_encoder_output_size(img_height: int, img_width: int) -> tuple:
    """
    Compute encoder output size after downsampling.
    
    The DenseNet encoder uses ceil_mode=True for pooling, so we need to
    account for that when computing output sizes.
    
    Downsampling chain:
    1. conv1 (stride=2): ceil(H/2), ceil(W/2)
    2. max_pool2d (2): ceil(H/4), ceil(W/4)  
    3. trans1 avg_pool2d (2): ceil(H/8), ceil(W/8)
    4. trans2 avg_pool2d (2): ceil(H/16), ceil(W/16)
    """
    h, w = img_height, img_width
    
    # 4 halving operations with ceil
    for _ in range(4):
        h = (h + 1) // 2
        w = (w + 1) // 2
    
    return max(h, 1), max(w, 1)


def extract_samples_from_zip(archive: ZipFile, folder: str):
    """Extract samples from a zip archive."""
    samples = []
    try:
        caption_path = f"data/{folder}/caption.txt"
        with archive.open(caption_path, 'r') as f:
            captions = f.readlines()
        
        for line in captions:
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
                samples.append((img_name, img, formula))
            except KeyError:
                continue
    except KeyError:
        print(f"Warning: Could not find {folder} in archive")
        return []
    return samples


def process_single_sample(args):
    """Process a single sample to generate ground truth maps."""
    (img_name, img_pil, latex, output_dir, gt_type, size_mode, 
     fixed_height, fixed_width, use_multilabel, dpi) = args
    
    try:
        img_w, img_h = img_pil.size
        
        # Determine target size based on mode
        if size_mode == 'dynamic':
            target_h, target_w = compute_encoder_output_size(img_h, img_w)
        else:  # 'fixed'
            target_h, target_w = fixed_height, fixed_width
        
        img_np = np.array(img_pil).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        
        result = {
            'img_height': img_h,
            'img_width': img_w,
            'target_height': target_h,
            'target_width': target_w,
            'size_mode': size_mode,
            'latex': latex,
            'multilabel': use_multilabel,
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate spatial map
        if gt_type in ['spatial', 'both']:
            spatial_gen = SpatialMapGenerator(
                img_height=img_h,
                img_width=img_w,
                target_height=target_h,
                target_width=target_w
            )
            spatial_map = spatial_gen(img_tensor).numpy()
            result['spatial_map'] = spatial_map
        
        # Generate relation map
        if gt_type in ['relation', 'both']:
            if use_multilabel:
                # Use multi-label generator with color-coded rendering
                relation_gen = RelationMapGenerator(
                    target_height=target_h,
                    target_width=target_w,
                    dpi=dpi,
                    use_gaussian=True
                )
                # Generate multi-label map [7, H, W]
                relation_map, symbol_infos, _, _ = relation_gen.generate(latex)
                result['relation_map'] = relation_map  # (7, H, W)
                
                # Also store symbol info for debugging
                result['num_symbols'] = len(symbol_infos)
        
        # Save to npz file
        npz_path = output_dir / f"{img_name}_gt.npz"
        np.savez_compressed(str(npz_path), **result)
        
        return (img_name, True, None, (img_h, img_w, target_h, target_w))
    except Exception as e:
        import traceback
        return (img_name, False, str(e) + "\n" + traceback.format_exc(), None)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate ground truth maps for CoMER multi-task training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate both spatial and relation maps with DYNAMIC size (matches encoder)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --size_mode dynamic
    
    # Generate with fixed size (for visualization)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --size_mode fixed
    
    # Generate only relation maps
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --type relation
        """
    )
    parser.add_argument('--data_zip', type=str, default='data.zip', help='Path to data.zip')
    parser.add_argument('--output_dir', type=str, default='data/cached_maps', help='Output directory')
    parser.add_argument('--split', type=str, default='all', 
                        choices=['train', '2014', '2016', '2019', 'all'],
                        help='Dataset split to process')
    parser.add_argument('--type', type=str, default='both',
                        choices=['spatial', 'relation', 'both'],
                        help='Type of ground truth to generate')
    parser.add_argument('--size_mode', type=str, default='dynamic',
                        choices=['fixed', 'dynamic'],
                        help='Size mode: fixed (32x64) or dynamic (based on encoder 16x downsample)')
    parser.add_argument('--fixed_height', type=int, default=DEFAULT_FIXED_HEIGHT,
                        help='Fixed target height (used when size_mode=fixed)')
    parser.add_argument('--fixed_width', type=int, default=DEFAULT_FIXED_WIDTH,
                        help='Fixed target width (used when size_mode=fixed)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--max_samples', type=int, default=-1, 
                        help='Max samples per split (-1 for all)')
    parser.add_argument('--no_multilabel', action='store_true',
                        help='Disable multi-label relation generation (use old method)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='DPI for LaTeX rendering (only for multi-label)')
    
    args = parser.parse_args()
    
    data_zip_path = Path(args.data_zip)
    if not data_zip_path.exists():
        print(f"Error: data.zip not found at {data_zip_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_multilabel = USING_MULTILABEL and not args.no_multilabel
    
    print("=" * 80)
    print("PRE-GENERATING GROUND TRUTH MAPS")
    print("=" * 80)
    print(f"Data zip: {data_zip_path}")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth type: {args.type.upper()}")
    print(f"Size mode: {args.size_mode.upper()}")
    
    if args.size_mode == 'fixed':
        print(f"  Fixed target size: {args.fixed_height}x{args.fixed_width}")
    else:
        print(f"  Encoder downsample factor: {ENCODER_DOWNSAMPLE_FACTOR}x")
        print(f"  Ground truth size = ceil(img_size / {ENCODER_DOWNSAMPLE_FACTOR})")
    
    if args.type in ['relation', 'both']:
        if use_multilabel:
            print(f"\n✓ Using MULTI-LABEL relation generator (color-coded rendering)")
            print(f"  Each symbol can have MULTIPLE relation labels")
            print(f"  Rendering DPI: {args.dpi}")
        else:
            print(f"\n⚠ Using single-label relation generator (handwriting-based)")
    
    if args.type == 'both':
        print("\nGenerating:")
        print("  - Spatial maps (S): Binary symbol region masks [1, H, W]")
        print(f"  - Relation maps (R): {RelationType.num_classes()}-class structure maps [7, H, W]")
        print(f"    Classes: {', '.join(RelationType.names())}")
    elif args.type == 'spatial':
        print("\nGenerating:")
        print("  - Spatial maps (S): Binary symbol region masks [1, H, W]")
    else:  # relation
        print("\nGenerating:")
        print(f"  - Relation maps (R): {RelationType.num_classes()}-class structure maps [7, H, W]")
        print(f"    Classes: {', '.join(RelationType.names())}")
    
    print("=" * 80)
    
    if args.split == 'all':
        splits = ['train', '2014', '2016', '2019']
    else:
        splits = [args.split]
    
    total_success = 0
    total_errors = 0
    all_sizes = []
    
    with ZipFile(data_zip_path, 'r') as archive:
        for split in splits:
            print(f"\nProcessing split: {split}")
            samples = extract_samples_from_zip(archive, split)
            if not samples:
                print(f"No samples found")
                continue
            
            if args.max_samples > 0:
                samples = samples[:args.max_samples]
            
            print(f"Found {len(samples)} samples")
            
            split_output_dir = output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            process_args = [
                (img_name, img_pil, latex, split_output_dir, args.type, 
                 args.size_mode, args.fixed_height, args.fixed_width,
                 use_multilabel, args.dpi)
                for img_name, img_pil, latex in samples
            ]
            
            if args.workers > 1 and not use_multilabel:
                # Multi-processing only works with non-multilabel (no rendering)
                from multiprocessing import Pool
                with Pool(args.workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_single_sample, process_args),
                        total=len(process_args),
                        desc=f"{split}"
                    ))
            else:
                # Single process (required for multilabel due to xelatex)
                results = []
                for proc_args in tqdm(process_args, desc=f"{split}"):
                    results.append(process_single_sample(proc_args))
            
            success = sum(1 for _, ok, _, _ in results if ok)
            errors = len(results) - success
            total_success += success
            total_errors += errors
            
            for _, ok, _, sizes in results:
                if ok and sizes:
                    all_sizes.append(sizes)
            
            print(f"Success: {success}/{len(results)}")
            if errors > 0:
                print(f"Errors: {errors}")
                for name, ok, err, _ in results:
                    if not ok:
                        print(f"  - {name}: {err[:200]}...")
                        break  # Only show first error
    
    if all_sizes:
        img_heights = [s[0] for s in all_sizes]
        img_widths = [s[1] for s in all_sizes]
        target_heights = [s[2] for s in all_sizes]
        target_widths = [s[3] for s in all_sizes]
        
        print("\n" + "=" * 80)
        print("SIZE STATISTICS")
        print("=" * 80)
        print(f"Image sizes:")
        print(f"  Height - min: {min(img_heights):4d}, max: {max(img_heights):4d}, avg: {np.mean(img_heights):.1f}")
        print(f"  Width  - min: {min(img_widths):4d}, max: {max(img_widths):4d}, avg: {np.mean(img_widths):.1f}")
        print(f"Ground truth sizes:")
        print(f"  Height - min: {min(target_heights):4d}, max: {max(target_heights):4d}, avg: {np.mean(target_heights):.1f}")
        print(f"  Width  - min: {min(target_widths):4d}, max: {max(target_widths):4d}, avg: {np.mean(target_widths):.1f}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Total success: {total_success}")
    print(f"Total errors: {total_errors}")
    print(f"Cached maps saved to: {output_dir}")
    print("=" * 80)
    
    # Print example of how to load the data
    print("\nExample usage to load cached ground truth:")
    print("```python")
    print("import numpy as np")
    print("data = np.load('data/cached_maps/train/sample_gt.npz')")
    if args.type in ['spatial', 'both']:
        print("spatial_map = data['spatial_map']  # Shape: (1, H, W)")
    if args.type in ['relation', 'both']:
        print("relation_map = data['relation_map']  # Shape: (7, H, W) - MULTI-LABEL")
    print("```")


if __name__ == '__main__':
    main()
