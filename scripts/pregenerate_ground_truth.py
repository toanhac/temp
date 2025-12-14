#!/usr/bin/env python3
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

from comer.datamodule.spatial_gt import AuxiliaryTargetGenerator

ENCODER_DOWNSAMPLE_FACTOR = 16


def extract_samples_from_zip(archive: ZipFile, folder: str):
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


def compute_encoder_output_size(img_height, img_width, factor=ENCODER_DOWNSAMPLE_FACTOR):
    h = img_height
    w = img_width
    for _ in range(4):
        h = (h + 1) // 2
        w = (w + 1) // 2
    return h, w


def process_single_sample(args):
    img_name, img_pil, latex, output_dir = args
    try:
        img_w, img_h = img_pil.size
        target_h, target_w = compute_encoder_output_size(img_h, img_w)
        target_h = max(target_h, 1)
        target_w = max(target_w, 1)
        
        generator = AuxiliaryTargetGenerator(
            img_height=img_h,
            img_width=img_w,
            target_height=target_h,
            target_width=target_w,
            device='cpu'
        )
        img_np = np.array(img_pil).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        S_target = generator(img_tensor, latex)
        spatial_map = S_target.numpy()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        spatial_path = output_dir / f"{img_name}_spatial.npy"
        
        metadata = {
            'img_height': img_h,
            'img_width': img_w,
            'target_height': target_h,
            'target_width': target_w,
        }
        
        np.savez_compressed(
            str(spatial_path).replace('.npy', '.npz'),
            spatial_map=spatial_map,
            img_height=img_h,
            img_width=img_w,
            target_height=target_h,
            target_width=target_w,
        )
        
        return (img_name, True, None, (img_h, img_w, target_h, target_w))
    except Exception as e:
        return (img_name, False, str(e), None)


def main():
    parser = argparse.ArgumentParser(description='Pre-generate spatial maps for CoMER training')
    parser.add_argument('--data_zip', type=str, default='data.zip', help='Path to data.zip')
    parser.add_argument('--output_dir', type=str, default='data/cached_maps', help='Output directory')
    parser.add_argument('--split', type=str, default='all', choices=['train', '2014', '2016', '2019', 'all'])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    
    data_zip_path = Path(args.data_zip)
    if not data_zip_path.exists():
        print(f"Error: data.zip not found at {data_zip_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PRE-GENERATING SPATIAL MAPS (Dynamic Size)")
    print("=" * 80)
    print(f"Data zip: {data_zip_path}")
    print(f"Output directory: {output_dir}")
    print(f"Encoder downsample factor: {ENCODER_DOWNSAMPLE_FACTOR}x")
    print("Note: Each spatial map size matches its image's encoder output size")
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
            print(f"Found {len(samples)} samples")
            
            split_output_dir = output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            process_args = [
                (img_name, img_pil, latex, split_output_dir)
                for img_name, img_pil, latex in samples
            ]
            
            if args.workers > 1:
                from multiprocessing import Pool
                with Pool(args.workers) as pool:
                    results = list(tqdm(pool.imap(process_single_sample, process_args), total=len(process_args), desc=f"{split}"))
            else:
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
                for name, ok, err, _ in results[:5]:
                    if not ok:
                        print(f"  - {name}: {err}")
    
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
        print(f"Spatial map sizes (encoder output):")
        print(f"  Height - min: {min(target_heights):4d}, max: {max(target_heights):4d}, avg: {np.mean(target_heights):.1f}")
        print(f"  Width  - min: {min(target_widths):4d}, max: {max(target_widths):4d}, avg: {np.mean(target_widths):.1f}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Total success: {total_success}")
    print(f"Total errors: {total_errors}")
    print(f"Cached maps saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
