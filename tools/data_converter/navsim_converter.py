# -*- coding: utf-8 -*-
import argparse
import mmcv
import numpy as np
import os
import os.path as osp
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from pathlib import Path

# python navsim_converter.py --data-root /home/byounggun/MapTR/data/navsim --out-dir /home/byounggun/MapTR/data/navsim

def process_pkl_file(pkl_path, data_root, split):
    try:
        with open(pkl_path, 'rb') as f:
            frames = pickle.load(f)
        
        if not frames: return []

        first_frame = frames[0]
        log_name = first_frame.get('log_name', pkl_path.stem)
        map_location = first_frame.get('map_location', None)
        
        if not map_location:
             # 맵 이름 추론
             pkl_str = str(pkl_path)
             if 'us-ma-boston' in pkl_str: map_location = 'us-ma-boston'
             elif 'us-nv-las-vegas' in pkl_str: map_location = 'us-nv-las-vegas-strip'
             elif 'pittsburgh' in pkl_str: map_location = 'us-pa-pittsburgh-hazelwood'
             elif 'sg-one-north' in pkl_str: map_location = 'sg-one-north'
             else: return []

        samples = []
        for frame in frames:
            token = frame['token']
            timestamp = frame['timestamp']
            
            try:
                e2g_trans = np.array(frame['ego2global_translation'])
                e2g_rot = np.array(frame['ego2global_rotation'])
            except KeyError:
                continue

            cams = {}
            if 'cams' in frame:
                raw_cams = frame['cams']
                for cam_name, cam_info in raw_cams.items():
                    if isinstance(cam_info, dict):
                        # 1. 이미지 절대 경로 생성
                        rel_path = cam_info.get('data_path', '')
                        if not rel_path: continue
                        
                        # Remove split prefix if exists to avoid duplication
                        if rel_path.startswith(split + '/'):
                            rel_path = rel_path[len(split)+1:]
                        
                        # sensor_blobs/{split}/{rel_path}
                        # 예: /data2/.../sensor_blobs/mini/2021.../CAM_F0/xxx.jpg
                        if split == 'test':
                             # test_sensor_blobs/test/...
                             abs_path = os.path.join(data_root, 'download', 'test_sensor_blobs', split, rel_path)
                        elif split == 'trainval':
                             # trainval_sensor_blobs/trainval/...
                             abs_path = os.path.join(data_root, 'download', 'trainval_sensor_blobs', split, rel_path)
                        else:
                             abs_path = os.path.join(data_root, 'sensor_blobs', split, rel_path)
                        
                        # 2. Extrinsics 계산 (sensor2lidar -> lidar2cam)
                        # MapTR은 lidar2cam (lidar -> camera 변환 행렬)을 사용함
                        if 'sensor2lidar_rotation' in cam_info:
                            s2l_r = cam_info['sensor2lidar_rotation']
                            s2l_t = cam_info['sensor2lidar_translation']
                            
                            # 4x4 행렬 생성 (Sensor -> Lidar)
                            sensor2lidar = np.eye(4)
                            sensor2lidar[:3, :3] = s2l_r
                            sensor2lidar[:3, 3] = s2l_t
                            
                            # 역행렬 계산 (Lidar -> Sensor/Camera)
                            try:
                                lidar2cam = np.linalg.inv(sensor2lidar)
                            except np.linalg.LinAlgError:
                                lidar2cam = np.eye(4)
                        else:
                            lidar2cam = np.eye(4) # fallback

                        cams[cam_name] = dict(
                            data_path=str(abs_path),
                            cam_intrinsic=cam_info.get('cam_intrinsic', np.eye(3)),
                            lidar2cam_rt=lidar2cam, # [중요] 변환된 Extrinsic 저장
                            # 아래는 MapTR 로더 호환성을 위해 남겨둠 (사용 안 할 수도 있음)
                            sensor2ego_translation=np.zeros(3),
                            sensor2ego_rotation=np.array([1,0,0,0]),
                        )
            
            lidar_path = frame.get('lidar_path', "")

            samples.append(dict(
                token=str(token),
                log_id=log_name,
                map_location=map_location,
                ego2global_translation=e2g_trans,
                ego2global_rotation=e2g_rot,
                cams=cams, 
                timestamp=timestamp,
                lidar_path=str(lidar_path), 
                sample_idx=str(token)
            ))
    except Exception as e:
        return []

    return samples

def create_navsim_infos(data_root, out_dir, split, nproc):
    print(f"Processing split: {split}")
    
    if split == 'test':
        logs_root = Path(data_root) / "download" / "test_navsim_logs" / split
    elif split == 'trainval':
        # Try multiple locations for trainval
        candidates = [
            Path(data_root) / "download" / "trainval_navsim_logs" / split,
            Path(data_root) / "download" / "mini_navsim_logs" / split,
            Path(data_root) / "navsim_logs" / split
        ]
        logs_root = None
        for cand in candidates:
            if cand.exists():
                logs_root = cand
                break
        
        if logs_root is None:
             # Default to one of them for error message
             logs_root = candidates[0]
    else:
        logs_root = Path(data_root) / "navsim_logs" / split
    if not logs_root.exists():
        print(f"Skipping {split} (path not found: {logs_root})")
        return

    pkl_files = list(logs_root.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} logs in {logs_root}")
    
    if len(pkl_files) == 0:
        print(f"No .pkl files found. Check directory.")
        return

    # split 정보를 process_pkl_file에 전달하기 위해 partial 사용
    process_func = partial(process_pkl_file, data_root=data_root, split=split)
    all_samples = []
    
    with Pool(nproc) as p:
        for samples in tqdm(p.imap_unordered(process_func, pkl_files), total=len(pkl_files)):
            all_samples.extend(samples)
    
    print(f"Extracted {len(all_samples)} samples.")
    
    if len(all_samples) == 0:
        return

    # NuScenes 스타일: 맵 데이터 없이 샘플 정보만 저장
    infos = dict(samples=all_samples)
    out_file = osp.join(out_dir, f"navsim_map_infos_{split}.pkl")
    mmcv.dump(infos, out_file)
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ['mini', 'trainval', 'test']:
    # for split in ['test']:
        create_navsim_infos(args.data_root, args.out_dir, split, args.nproc)