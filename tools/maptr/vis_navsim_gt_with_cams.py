# -*- coding: utf-8 -*-
"""
NavSim GT + Camera 시각화 도구
카메라 8개 + GT BEV Map을 함께 시각화

Usage:
    cd /home/byounggun/MapTR
    python tools/maptr/vis_navsim_gt_with_cams.py \
        projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
        --show-dir ./vis_navsim_combined \
        --num-samples 5
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import mmcv
import torch
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# NavSim 8개 카메라
NAVSIM_CAMS = ['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0']

# Map class colors (BGR for OpenCV)
COLORS_CV2 = {
    0: (0, 165, 255),   # divider - orange
    1: (255, 0, 0),     # ped_crossing - blue
    2: (0, 255, 0),     # boundary - green
}
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim GT + Camera 시각화')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--show-dir', default='./vis_navsim_combined', help='output directory')
    parser.add_argument('--num-samples', type=int, default=5, help='number of samples')
    args = parser.parse_args()
    return args


def draw_gt_on_bev(gt_bboxes_3d, gt_labels_3d, pc_range, img_size=800):
    """GT map을 OpenCV 이미지로 그리기"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 좌표 변환 함수: ego coords -> image coords
    # ego: X=forward, Y=left
    # image: 위=forward, 왼쪽=left
    def ego_to_img(ego_x, ego_y):
        # X(forward) -> img_y (위가 forward이므로 뒤집기)
        # Y(left) -> img_x (왼쪽이 left)
        scale = img_size / (pc_range[3] - pc_range[0])
        img_x = int(((-ego_y) - pc_range[1]) * scale)  # -Y -> right is positive
        img_y = int((pc_range[3] - ego_x) * scale)     # flip X for image coords
        return img_x, img_y
    
    num_elements = {0: 0, 1: 0, 2: 0}
    
    try:
        if hasattr(gt_bboxes_3d, 'fixed_num_sampled_points'):
            gt_lines = gt_bboxes_3d.fixed_num_sampled_points
        elif hasattr(gt_bboxes_3d, '_fixed_pts'):
            gt_lines = gt_bboxes_3d._fixed_pts
        else:
            return img, num_elements
        
        for pts, label in zip(gt_lines, gt_labels_3d):
            if isinstance(pts, torch.Tensor):
                pts = pts.numpy()
            
            label_idx = int(label)
            if label_idx < 0 or label_idx > 2:
                continue
            
            # 유효한 점만 필터링
            valid_mask = (pts[:, 0] > -9000) & (pts[:, 1] > -9000)
            pts = pts[valid_mask]
            
            if len(pts) < 2:
                continue
            
            # 이미지 좌표로 변환
            img_pts = np.array([ego_to_img(p[0], p[1]) for p in pts])
            
            # 선 그리기
            color = COLORS_CV2[label_idx]
            for i in range(len(img_pts) - 1):
                pt1 = tuple(img_pts[i])
                pt2 = tuple(img_pts[i + 1])
                cv2.line(img, pt1, pt2, color, 2)
            
            # 점 그리기
            for pt in img_pts:
                cv2.circle(img, tuple(pt), 3, color, -1)
            
            num_elements[label_idx] += 1
    except Exception as e:
        print(f"Error drawing GT: {e}")
    
    # 차량 그리기 (중앙)
    center_x, center_y = ego_to_img(0, 0)
    car_pts = np.array([
        ego_to_img(2.5, -1),
        ego_to_img(2.5, 1),
        ego_to_img(-2.5, 1),
        ego_to_img(-2.5, -1),
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_pts], (128, 128, 128))
    
    # 전방 표시
    front_pt = ego_to_img(4, 0)
    cv2.arrowedLine(img, (center_x, center_y), front_pt, (0, 0, 255), 3, tipLength=0.3)
    
    # 범례
    y_offset = 30
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_CV2[i]
        cv2.rectangle(img, (10, y_offset + i*25), (30, y_offset + i*25 + 15), color, -1)
        cv2.putText(img, f"{name} ({num_elements[i]})", (40, y_offset + i*25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 타이틀
    cv2.putText(img, "GT Map (BEV)", (img_size//2 - 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img, num_elements


def create_camera_grid(data, sensor_root):
    """8개 카메라를 3x3 그리드로 배치 (중앙 빈칸)"""
    img_metas = data['img_metas']
    if hasattr(img_metas, 'data'):
        img_metas = img_metas.data
    
    # img_metas가 dict인 경우 (key가 0)
    if isinstance(img_metas, dict):
        img_metas = img_metas.get(0, img_metas)
    # list인 경우
    elif isinstance(img_metas, list) and len(img_metas) > 0:
        img_metas = img_metas[0]
        if isinstance(img_metas, list) and len(img_metas) > 0:
            img_metas = img_metas[0]
    
    cam_images = {}
    if isinstance(img_metas, dict) and 'filename' in img_metas:
        filenames = img_metas['filename']
        for i, cam_name in enumerate(NAVSIM_CAMS):
            if i < len(filenames):
                img_path = filenames[i]
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        cam_images[cam_name] = img
    
    if not cam_images:
        return None
    
    # 이미지 크기 조정
    target_h, target_w = 270, 480  # 각 카메라 이미지 크기
    
    def resize_cam(img, cam_name):
        img = cv2.resize(img, (target_w, target_h))
        # 카메라 이름 추가
        cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        return img
    
    # 레이아웃:
    # Row 1: L2, L1, L0, F0, R0, R1, R2
    # Row 2: (빈칸), (빈칸), (빈칸), B0, (빈칸), (빈칸), (빈칸)
    # 
    # 또는 더 컴팩트하게:
    # Row 1: L0, F0, R0
    # Row 2: L1, B0, R1
    # Row 3: L2, (info), R2
    
    row1_cams = ['CAM_L0', 'CAM_F0', 'CAM_R0']
    row2_cams = ['CAM_L1', 'CAM_B0', 'CAM_R1']
    row3_cams = ['CAM_L2', None, 'CAM_R2']
    
    def make_row(cam_list):
        row_imgs = []
        for cam in cam_list:
            if cam is None:
                # Info 패널
                info_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 50
                cv2.putText(info_img, "NavSim", (target_w//2-50, target_h//2-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(info_img, "8 Cameras", (target_w//2-60, target_h//2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                row_imgs.append(info_img)
            elif cam in cam_images:
                row_imgs.append(resize_cam(cam_images[cam], cam))
            else:
                # 빈 이미지
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(blank, cam + " (missing)", (10, target_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                row_imgs.append(blank)
        return cv2.hconcat(row_imgs) if row_imgs else None
    
    row1 = make_row(row1_cams)
    row2 = make_row(row2_cams)
    row3 = make_row(row3_cams)
    
    if row1 is not None and row2 is not None and row3 is not None:
        return cv2.vconcat([row1, row2, row3])
    
    return None


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
            plg_lib = importlib.import_module(_module_path)
    
    # Build dataset
    dataset = build_dataset(cfg.data.train)
    
    os.makedirs(args.show_dir, exist_ok=True)
    
    sensor_root = cfg.data.train.get('sensor_root', '')
    
    print(f"\n{'='*60}")
    print(f"Generating {args.num_samples} combined visualizations")
    print(f"Output: {args.show_dir}")
    print(f"{'='*60}\n")
    
    prog_bar = mmcv.ProgressBar(args.num_samples)
    
    for idx in range(min(args.num_samples, len(dataset))):
        try:
            data = dataset[idx]
            
            # GT 추출
            gt_labels = data.get('gt_labels_3d')
            gt_bboxes = data.get('gt_bboxes_3d')
            
            if gt_labels is None or gt_bboxes is None:
                prog_bar.update()
                continue
            
            if hasattr(gt_labels, 'data'):
                gt_labels = gt_labels.data
            if hasattr(gt_bboxes, 'data'):
                gt_bboxes = gt_bboxes.data
            
            # GT BEV 그리기
            pc_range = cfg.point_cloud_range
            gt_img, num_elements = draw_gt_on_bev(gt_bboxes, gt_labels, pc_range, img_size=810)
            
            # 카메라 그리드 생성
            cam_grid = create_camera_grid(data, sensor_root)
            
            if cam_grid is None:
                prog_bar.update()
                continue
            
            # GT 이미지 크기 맞추기
            cam_h, cam_w = cam_grid.shape[:2]
            gt_img_resized = cv2.resize(gt_img, (cam_h, cam_h))  # 정사각형 유지
            
            # 합치기: 카메라 | GT BEV
            combined = cv2.hconcat([cam_grid, gt_img_resized])
            
            # 저장
            sample_dir = os.path.join(args.show_dir, f'sample_{idx:05d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(sample_dir, 'combined.jpg'), combined, 
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(sample_dir, 'gt_bev.jpg'), gt_img,
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # 개별 카메라 저장
            img_metas = data['img_metas']
            if hasattr(img_metas, 'data'):
                img_metas = img_metas.data
            if isinstance(img_metas, dict):
                img_metas = img_metas.get(0, img_metas)
            elif isinstance(img_metas, list) and len(img_metas) > 0:
                img_metas = img_metas[0]
                if isinstance(img_metas, list) and len(img_metas) > 0:
                    img_metas = img_metas[0]
            
            if isinstance(img_metas, dict) and 'filename' in img_metas:
                for i, cam_name in enumerate(NAVSIM_CAMS):
                    if i < len(img_metas['filename']):
                        src_path = img_metas['filename'][i]
                        if os.path.exists(src_path):
                            img = cv2.imread(src_path)
                            if img is not None:
                                cv2.imwrite(os.path.join(sample_dir, f'{cam_name}.jpg'), img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
        
        prog_bar.update()
    
    print(f"\n\n{'='*60}")
    print(f"Done! Output saved to: {args.show_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
