# -*- coding: utf-8 -*-
"""
NavSim GT Map Element 시각화 도구
MapTR 데이터 로딩이 올바르게 되고 있는지 확인하기 위한 스크립트

Usage:
    cd /home/byounggun/MapTR
    python tools/maptr/vis_navsim_gt.py \
        projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
        --show-dir ./vis_navsim_gt \
        --num-samples 10

Author: Based on vis_pred.py, adapted for NavSim
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import mmcv
import shutil
import torch
import warnings
import numpy as np
from mmcv import Config
from mmdet3d.utils import get_root_logger
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import replace_ImageToTensor
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# NavSim에서는 8개의 카메라를 사용
NAVSIM_CAMS = [
    'CAM_F0',  # Front 
    'CAM_L0',  # Left Front
    'CAM_L1',  # Left 
    'CAM_L2',  # Left Rear
    'CAM_R0',  # Right Front
    'CAM_R1',  # Right
    'CAM_R2',  # Right Rear
    'CAM_B0',  # Back
]

# NuScenes 카메라 이름 (호환성)
NUSCENES_CAMS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]

# Map class colors: divider->orange, ped_crossing->blue, boundary->green
COLORS_PLT = ['orange', 'b', 'g']
COLORS_CV2 = [(0, 165, 255), (255, 0, 0), (0, 255, 0)]  # BGR for OpenCV


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim GT Map Element 시각화')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--show-dir', default='./vis_navsim_gt', help='directory where visualizations will be saved')
    parser.add_argument('--num-samples', type=int, default=20, help='number of samples to visualize')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='which split to visualize')
    parser.add_argument('--show-cam', action='store_true', help='show camera images')
    parser.add_argument('--skip-empty', action='store_true', default=True, help='skip samples without GT')
    parser.add_argument('--show-stats', action='store_true', default=True, help='show dataset statistics')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts'],
        help='GT vis format: "fixed_num_pts", "polyline_pts", "bbox"')
    args = parser.parse_args()
    return args


def create_surrounding_view(sample_dir, cam_names, logger):
    """카메라 이미지들을 합쳐서 surroud view 생성"""
    cam_images = {}
    for cam in cam_names:
        cam_img_path = osp.join(sample_dir, cam + '.jpg')
        if osp.exists(cam_img_path):
            cam_images[cam] = cv2.imread(cam_img_path)
    
    if len(cam_images) == 0:
        return None
    
    # NavSim 8 camera layout
    if len(cam_images) >= 8:
        # Front row: L0, F0, R0
        row_1 = []
        for cam in ['CAM_L0', 'CAM_F0', 'CAM_R0']:
            if cam in cam_images:
                row_1.append(cam_images[cam])
        
        # Back row: L2, B0, R2
        row_2 = []
        for cam in ['CAM_L2', 'CAM_B0', 'CAM_R2']:
            if cam in cam_images:
                row_2.append(cam_images[cam])
        
        if len(row_1) > 0 and len(row_2) > 0:
            try:
                row_1_img = cv2.hconcat(row_1)
                row_2_img = cv2.hconcat(row_2)
                if row_1_img.shape[1] != row_2_img.shape[1]:
                    # Resize to match width
                    target_w = min(row_1_img.shape[1], row_2_img.shape[1])
                    row_1_img = cv2.resize(row_1_img, (target_w, row_1_img.shape[0]))
                    row_2_img = cv2.resize(row_2_img, (target_w, row_2_img.shape[0]))
                return cv2.vconcat([row_1_img, row_2_img])
            except Exception as e:
                logger.warning(f"Failed to create surround view: {e}")
    
    # NuScenes 6 camera layout fallback
    elif len(cam_images) >= 6:
        row_1 = []
        for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']:
            if cam in cam_images:
                row_1.append(cam_images[cam])
        row_2 = []
        for cam in ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
            if cam in cam_images:
                row_2.append(cam_images[cam])
        
        if len(row_1) == 3 and len(row_2) == 3:
            row_1_img = cv2.hconcat(row_1)
            row_2_img = cv2.hconcat(row_2)
            return cv2.vconcat([row_1_img, row_2_img])
    
    return None


def visualize_gt_on_bev(gt_bboxes_3d, gt_labels_3d, pc_range, save_path, vis_format='fixed_num_pts', car_img=None):
    """GT map elements를 BEV에 시각화
    
    NuScenes/NavSim Lidar 좌표계:
    - X축: 전방 (forward +)
    - Y축: 왼쪽 (left +)
    
    시각화에서는 X(전방)가 위쪽, Y(왼쪽)가 왼쪽으로 표시
    즉, plot(y, x) 또는 plot(-y, x)로 변환
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 시각화: 수평축 = -Y (오른쪽이 +), 수직축 = X (위쪽이 +)
    # pc_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    ax.set_xlim(-pc_range[4], -pc_range[1])  # -Y range (right is positive)
    ax.set_ylim(pc_range[0], pc_range[3])    # X range (front is up)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'GT Map Elements ({vis_format})', fontsize=12)
    
    num_elements = {'divider': 0, 'ped_crossing': 0, 'boundary': 0}
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    if vis_format == 'fixed_num_pts':
        # gt_bboxes_3d is LiDARInstanceLines object
        try:
            if hasattr(gt_bboxes_3d, '_fixed_pts'):
                # Pre-generated data
                gt_lines_fixed_num_pts = gt_bboxes_3d._fixed_pts
            else:
                gt_lines_fixed_num_pts = gt_bboxes_3d.fixed_num_sampled_points
            
            for gt_pts, gt_label in zip(gt_lines_fixed_num_pts, gt_labels_3d):
                if isinstance(gt_pts, torch.Tensor):
                    pts = gt_pts.numpy()
                else:
                    pts = gt_pts
                
                label_idx = int(gt_label)
                if label_idx < 0 or label_idx >= len(class_names):
                    continue
                    
                ego_x = pts[:, 0]  # forward
                ego_y = pts[:, 1]  # left
                
                # Skip padding values
                valid_mask = (ego_x > -9000) & (ego_y > -9000)
                if not valid_mask.any():
                    continue
                
                ego_x = ego_x[valid_mask]
                ego_y = ego_y[valid_mask]
                
                # Transform to visualization: plot_x = -ego_y, plot_y = ego_x
                plot_x = -ego_y
                plot_y = ego_x
                
                ax.plot(plot_x, plot_y, color=COLORS_PLT[label_idx], linewidth=1.5, alpha=0.8, zorder=1)
                ax.scatter(plot_x, plot_y, color=COLORS_PLT[label_idx], s=8, alpha=0.8, zorder=2)
                
                num_elements[class_names[label_idx]] += 1
                
        except Exception as e:
            print(f"Error visualizing fixed_num_pts: {e}")
    
    elif vis_format == 'polyline_pts':
        try:
            gt_lines_instance = gt_bboxes_3d.instance_list
            for gt_line, gt_label in zip(gt_lines_instance, gt_labels_3d):
                pts = np.array(list(gt_line.coords))
                label_idx = int(gt_label)
                if label_idx < 0 or label_idx >= len(class_names):
                    continue
                
                ego_x = pts[:, 0]  # forward
                ego_y = pts[:, 1]  # left
                
                # Transform to visualization: plot_x = -ego_y, plot_y = ego_x
                plot_x = -ego_y
                plot_y = ego_x
                
                ax.plot(plot_x, plot_y, color=COLORS_PLT[label_idx], linewidth=1.5, alpha=0.8, zorder=1)
                ax.scatter(plot_x, plot_y, color=COLORS_PLT[label_idx], s=8, alpha=0.8, zorder=2)
                
                num_elements[class_names[label_idx]] += 1
        except Exception as e:
            print(f"Error visualizing polyline_pts: {e}")
    
    elif vis_format == 'bbox':
        try:
            gt_lines_bbox = gt_bboxes_3d.bbox
            for gt_bbox, gt_label in zip(gt_lines_bbox, gt_labels_3d):
                if isinstance(gt_bbox, torch.Tensor):
                    gt_bbox = gt_bbox.numpy()
                label_idx = int(gt_label)
                if label_idx < 0 or label_idx >= len(class_names):
                    continue
                
                # bbox: [x_min, y_min, x_max, y_max] in ego coords
                # Transform: plot_x = -ego_y, plot_y = ego_x
                plot_x = -gt_bbox[3]  # -y_max
                plot_y = gt_bbox[0]   # x_min
                width = gt_bbox[3] - gt_bbox[1]   # y_max - y_min
                height = gt_bbox[2] - gt_bbox[0]  # x_max - x_min
                ax.add_patch(Rectangle((plot_x, plot_y), width, height, linewidth=1, 
                                              edgecolor=COLORS_PLT[label_idx], facecolor='none'))
                num_elements[class_names[label_idx]] += 1
        except Exception as e:
            print(f"Error visualizing bbox: {e}")
    
    # Add car icon at center (car facing up)
    if car_img is not None:
        ax.imshow(car_img, extent=[-1.5, 1.5, -2.0, 2.0], zorder=10)
    else:
        # Draw simple car shape: rectangle + triangle for direction
        car_body = plt.Rectangle((-0.9, -2.0), 1.8, 4.0, color='gray', alpha=0.8, zorder=10)
        car_front = plt.Polygon([[-0.9, 2.0], [0.9, 2.0], [0, 3.0]], color='red', alpha=0.8, zorder=11)
        ax.add_patch(car_body)
        ax.add_patch(car_front)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='orange', lw=2, label=f'divider ({num_elements["divider"]})'),
        plt.Line2D([0], [0], color='b', lw=2, label=f'ped_crossing ({num_elements["ped_crossing"]})'),
        plt.Line2D([0], [0], color='g', lw=2, label=f'boundary ({num_elements["boundary"]})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add axis labels for clarity
    ax.set_xlabel('← Left    Right →', fontsize=8)
    ax.set_ylabel('← Back    Front →', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.axis('on')
    
    plt.savefig(save_path, bbox_inches='tight', format='png', dpi=200)
    plt.close()
    
    return num_elements


def print_dataset_stats(dataset, logger):
    """데이터셋 통계 출력"""
    logger.info("="*60)
    logger.info("NavSim Dataset Statistics")
    logger.info("="*60)
    logger.info(f"Total samples: {len(dataset)}")
    
    # Sample a few to check data format
    sample_info = dataset.data_infos[0] if len(dataset.data_infos) > 0 else {}
    logger.info(f"Sample info keys: {list(sample_info.keys())}")
    
    if 'map_location' in sample_info:
        locations = set(info['map_location'] for info in dataset.data_infos)
        logger.info(f"Map locations: {locations}")
    
    if 'cams' in sample_info:
        cam_names = list(sample_info['cams'].keys())
        logger.info(f"Camera names: {cam_names}")
    
    logger.info("="*60)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Import plugins
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"Loading plugin: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

    # Select which dataset split to use
    if args.split == 'train':
        data_cfg = cfg.data.train.copy()
    elif args.split == 'val':
        data_cfg = cfg.data.val.copy()
    else:
        data_cfg = cfg.data.test.copy()
    
    # Keep test_mode=False to get GT labels in the data dict
    data_cfg.test_mode = False
    samples_per_gpu = 1
    
    if samples_per_gpu > 1:
        data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)

    # Create output directory
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    
    logger = get_root_logger()
    logger.info(f"Output directory: {args.show_dir}")
    logger.info(f"Visualizing {args.split} split")

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(data_cfg)
    dataset.is_vis_on_test = True
    
    if args.show_stats:
        print_dataset_stats(dataset, logger)
    
    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,  # Use 0 for debugging
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info(f'Dataset built: {len(dataset)} samples')

    # Get normalization params
    img_norm_cfg = cfg.img_norm_cfg
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)

    # Get pc_range
    pc_range = cfg.point_cloud_range

    # Load car icon
    car_img = None
    try:
        car_img = Image.open('./figs/lidar_car.png')
    except:
        logger.warning("Car icon not found, skipping")

    logger.info('='*60)
    logger.info('Starting NavSim GT visualization')
    logger.info(f'Number of samples to visualize: {min(args.num_samples, len(dataset))}')
    logger.info('='*60)

    # Statistics
    total_elements = {'divider': 0, 'ped_crossing': 0, 'boundary': 0}
    samples_with_gt = 0
    samples_without_gt = 0

    prog_bar = mmcv.ProgressBar(min(args.num_samples, len(dataset)))
    visualized = 0
    
    for i, data in enumerate(data_loader):
        if visualized >= args.num_samples:
            break
        
        # Debug: print available keys on first iteration
        if i == 0:
            logger.info(f"Data keys: {list(data.keys())}")
        
        # Check for GT labels - handle both train and test mode data format
        gt_labels_3d = None
        gt_bboxes_3d = None
        
        if 'gt_labels_3d' in data:
            gt_labels_3d = data['gt_labels_3d'].data[0]
            gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        else:
            # Test mode may not have GT in data dict, need to get from dataset directly
            # Get the sample index and load GT manually
            try:
                idx = i  # Current sample index
                input_dict = dataset.get_data_info(idx)
                dataset.pre_pipeline(input_dict)
                example = dataset.pipeline(input_dict)
                example = dataset.vectormap_pipeline(example, input_dict)
                
                if example is not None and 'gt_labels_3d' in example:
                    gt_labels_3d = [example['gt_labels_3d'].data]
                    gt_bboxes_3d = [example['gt_bboxes_3d'].data]
                else:
                    logger.warning(f"Sample {i}: No GT available after pipeline")
                    samples_without_gt += 1
                    continue
            except Exception as e:
                logger.warning(f"Sample {i}: Failed to load GT: {e}")
                samples_without_gt += 1
                continue
        
        # Check if this sample has valid GT
        has_valid_gt = False
        if len(gt_labels_3d) > 0:
            if isinstance(gt_labels_3d[0], torch.Tensor):
                has_valid_gt = (gt_labels_3d[0] != -1).any()
            else:
                has_valid_gt = len(gt_labels_3d[0]) > 0
        
        if args.skip_empty and not has_valid_gt:
            samples_without_gt += 1
            continue
        
        samples_with_gt += 1
        
        # Handle DataContainer for img_metas
        img_metas_dc = data['img_metas']
        if hasattr(img_metas_dc, 'data'):
            img_metas = img_metas_dc.data
        else:
            img_metas = img_metas_dc
        
        # img_metas can be a dict (queue format) or list
        if isinstance(img_metas, dict):
            # Queue format: {0: meta0, 1: meta1, ...}
            # Get the last one (current frame)
            last_key = max(img_metas.keys())
            meta = img_metas[last_key]
        elif isinstance(img_metas, list) and len(img_metas) > 0:
            meta = img_metas[0]
            # If meta is still a list, get first element
            if isinstance(meta, list) and len(meta) > 0:
                meta = meta[0]
        else:
            meta = {}
        
        # Ensure meta is a dict
        if not isinstance(meta, dict):
            logger.warning(f"Sample {i}: meta is not a dict, type={type(meta)}")
            meta = {}
        
        sample_idx = meta.get('sample_idx', f'sample_{i:05d}')
        if isinstance(sample_idx, str):
            sample_idx = sample_idx.replace('/', '_')[:50]  # Truncate long names
        
        sample_dir = osp.join(args.show_dir, str(sample_idx))
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))
        
        # Save camera images
        if args.show_cam and 'filename' in meta:
            filename_list = meta['filename']
            for filepath in filename_list:
                if not osp.exists(filepath):
                    continue
                filename = osp.basename(filepath)
                # Extract camera name from path
                cam_name = None
                for cam in NAVSIM_CAMS + NUSCENES_CAMS:
                    if cam.lower() in filepath.lower():
                        cam_name = cam
                        break
                if cam_name is None:
                    cam_name = osp.splitext(filename)[0]
                
                img_path = osp.join(sample_dir, cam_name + '.jpg')
                try:
                    shutil.copyfile(filepath, img_path)
                except Exception as e:
                    logger.warning(f"Failed to copy {filepath}: {e}")
            
            # Create surround view
            surround_img = create_surrounding_view(sample_dir, NAVSIM_CAMS + NUSCENES_CAMS, logger)
            if surround_img is not None:
                cv2.imwrite(osp.join(sample_dir, 'surround_view.jpg'), surround_img, 
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Visualize GT in each format
        for vis_format in args.gt_format:
            gt_map_path = osp.join(sample_dir, f'GT_{vis_format}_MAP.png')
            
            if has_valid_gt:
                num_elem = visualize_gt_on_bev(
                    gt_bboxes_3d[0], 
                    gt_labels_3d[0],
                    pc_range, 
                    gt_map_path,
                    vis_format=vis_format,
                    car_img=car_img
                )
                for k, v in num_elem.items():
                    total_elements[k] += v
            else:
                # Save empty figure
                plt.figure(figsize=(4, 8))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.title('No GT Map Elements', fontsize=10)
                plt.axis('off')
                if car_img is not None:
                    plt.imshow(car_img, extent=[-1.5, 1.5, -2.0, 2.0])
                plt.savefig(gt_map_path, bbox_inches='tight', format='png', dpi=300)
                plt.close()
        
        # Save sample info
        info_path = osp.join(sample_dir, 'info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Sample Index: {i}\n")
            f.write(f"Sample Token: {sample_idx}\n")
            if 'map_location' in meta:
                f.write(f"Map Location: {meta['map_location']}\n")
            if 'scene_token' in meta:
                f.write(f"Scene Token: {meta['scene_token']}\n")
            f.write(f"Num GT Labels: {len(gt_labels_3d[0]) if len(gt_labels_3d) > 0 else 0}\n")
            
            # Count elements per class
            if has_valid_gt:
                label_counts = {}
                for lbl in gt_labels_3d[0]:
                    lbl_val = int(lbl) if isinstance(lbl, (int, torch.Tensor)) else lbl
                    label_counts[lbl_val] = label_counts.get(lbl_val, 0) + 1
                f.write(f"Label counts: {label_counts}\n")
        
        visualized += 1
        prog_bar.update()

    # Print summary
    logger.info('\n' + '='*60)
    logger.info('Visualization Summary')
    logger.info('='*60)
    logger.info(f'Samples visualized: {visualized}')
    logger.info(f'Samples with GT: {samples_with_gt}')
    logger.info(f'Samples without GT (skipped): {samples_without_gt}')
    logger.info(f'Total GT elements:')
    for cls_name, count in total_elements.items():
        logger.info(f'  {cls_name}: {count}')
    logger.info(f'Output saved to: {args.show_dir}')
    logger.info('='*60)


if __name__ == '__main__':
    main()
