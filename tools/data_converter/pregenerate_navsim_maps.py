#!/usr/bin/env python
"""
Pre-generate map vectors for NavSim dataset to speed up training.
This eliminates the runtime map querying bottleneck (~11s per iteration).

Usage:
    python tools/data_converter/pregenerate_navsim_maps.py \
        --input-pkl data/navsim/navsim_map_infos_trainval_filtered.pkl \
        --output-pkl data/navsim/navsim_map_infos_trainval_with_maps.pkl \
        --data-root data/navsim \
        --nproc 16
"""

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from pyquaternion import Quaternion

# Import VectorizedLocalMap from dataset code
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from projects.mmdet3d_plugin.datasets.navsim_map_dataset import VectorizedLocalMap


def process_sample(sample, vector_map, patch_size):
    """Process a single sample to generate map vectors."""
    try:
        location = sample['map_location']
        e2g_t = sample['ego2global_translation']
        e2g_r = sample['ego2global_rotation']
        l2e_t = sample.get('lidar2ego_translation', np.zeros(3))
        l2e_r = sample.get('lidar2ego_rotation', np.array([1,0,0,0]))

        # Compute Lidar2Global
        T_global_ego = np.eye(4)
        T_global_ego[:3, :3] = Quaternion(e2g_r).rotation_matrix
        T_global_ego[:3, 3] = e2g_t

        T_ego_lidar = np.eye(4)
        T_ego_lidar[:3, :3] = Quaternion(l2e_r).rotation_matrix
        T_ego_lidar[:3, 3] = l2e_t

        T_global_lidar = T_global_ego @ T_ego_lidar
        
        l2g_t = T_global_lidar[:3, 3]
        l2g_r = Quaternion(matrix=T_global_lidar[:3, :3]).elements

        # Generate map vectors
        anns_results = vector_map.gen_vectorized_samples(location, l2g_t, l2g_r)
        
        # Extract the data we need
        gt_vecs_label = anns_results['gt_vecs_label']
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        
        # Get fixed num sampled points and shift
        if len(gt_vecs_pts_loc.instance_list) > 0:
            fixed_pts = gt_vecs_pts_loc.fixed_num_sampled_points.numpy()
            shift_pts = gt_vecs_pts_loc.shift_fixed_num_sampled_points_v2.numpy()
        else:
            fixed_pts = np.zeros((0, gt_vecs_pts_loc.fixed_num, 2), dtype=np.float32)
            shift_pts = np.zeros((0, 0, gt_vecs_pts_loc.fixed_num, 2), dtype=np.float32)
        
        # Add map data to sample
        sample['map_gt_labels'] = np.array(gt_vecs_label)
        sample['map_gt_pts_loc'] = fixed_pts
        sample['map_gt_pts_loc_shift'] = shift_pts
        sample['map_available'] = True
        
        return sample
        
    except Exception as e:
        # If map generation fails, mark as unavailable
        sample['map_available'] = False
        sample['map_error'] = str(e)
        return sample


def worker_init(data_root, patch_size, map_classes, fixed_num):
    """Initialize worker with VectorizedLocalMap (once per worker)."""
    global vector_map_global
    map_root = os.path.join(data_root, 'download', 'maps')
    if not os.path.exists(map_root):
        map_root = os.path.join(data_root, 'maps')
    
    vector_map_global = VectorizedLocalMap(
        dataroot=data_root,
        patch_size=patch_size,
        map_classes=map_classes,
        fixed_ptsnum_per_line=fixed_num,
        map_root=map_root
    )


def worker_process(sample):
    """Worker process function."""
    return process_sample(sample, vector_map_global, vector_map_global.patch_size)


def main():
    parser = argparse.ArgumentParser(description='Pre-generate map vectors for NavSim dataset')
    parser.add_argument('--input-pkl', type=str, required=True)
    parser.add_argument('--output-pkl', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    parser.add_argument('--fixed-num', type=int, default=20)
    args = parser.parse_args()

    # Calculate patch size
    patch_h = args.pc_range[4] - args.pc_range[1]
    patch_w = args.pc_range[3] - args.pc_range[0]
    patch_size = (patch_h, patch_w)
    
    map_classes = ('divider', 'ped_crossing', 'boundary')
    
    print(f"Loading input PKL: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    
    # Process in parallel
    print(f"Generating map vectors with {args.nproc} processes...")
    
    init_func = partial(worker_init, 
                       data_root=args.data_root,
                       patch_size=patch_size,
                       map_classes=map_classes,
                       fixed_num=args.fixed_num)
    
    with Pool(processes=args.nproc, initializer=init_func) as pool:
        processed_samples = list(tqdm(
            pool.imap(worker_process, samples),
            total=len(samples),
            desc="Processing samples"
        ))
    
    # Statistics
    available = sum(1 for s in processed_samples if s.get('map_available', False))
    print(f"\nStatistics:")
    print(f"  Total samples: {len(processed_samples)}")
    print(f"  With maps: {available}")
    print(f"  Without maps: {len(processed_samples) - available}")
    
    # Save output
    output_data = {'samples': processed_samples}
    print(f"\nSaving to {args.output_pkl}...")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("Done!")


if __name__ == '__main__':
    main()
