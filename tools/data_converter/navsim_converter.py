# -*- coding: utf-8 -*-
import argparse
import mmcv
import numpy as np
import os
import os.path as osp
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion
from multiprocessing import Pool
from functools import partial
from pathlib import Path

# [핵심] nuPlan API 대신 geopandas 직접 사용
try:
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
except ImportError:
    print("ERROR: geopandas가 설치되지 않았습니다. 'pip install geopandas'를 실행하세요.")
    exit()

# 실제 테이블 이름 매핑
NUPLAN_TO_MAPTR = {
    'divider': ['lanes_polygons', 'lane_connectors'],
    'ped_crossing': ['crosswalks'],
    'boundary': ['boundaries', 'generic_drivable_areas'], 
}

def get_map_vectors_from_city(maps_root, map_version, location):
    """
    geopandas를 사용하여 .gpkg 파일에서 직접 벡터 데이터를 추출합니다.
    """
    print(f"[{location}] 맵 벡터 추출 시작...")
    
    map_dir = os.path.join(maps_root, location)
    if not os.path.exists(map_dir):
        print(f"Warning: 맵 폴더를 찾을 수 없습니다: {map_dir}")
        return {}
    
    subdirs = sorted([d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))])
    if not subdirs:
        print(f"Warning: {location} 안에 버전 폴더가 없습니다.")
        return {}
    
    target_version = subdirs[-1] 
    gpkg_path = os.path.join(map_dir, target_version, "map.gpkg")
    
    if not os.path.exists(gpkg_path):
        return {}

    map_elements = {cls: [] for cls in NUPLAN_TO_MAPTR.keys()}
    
    for class_name, table_names in NUPLAN_TO_MAPTR.items():
        for table_name in table_names:
            try:
                gdf = gpd.read_file(gpkg_path, layer=table_name)
                if gdf.empty: continue

                for geom in gdf.geometry:
                    if geom is None: continue

                    pts_list = []
                    if geom.geom_type == 'LineString':
                        pts_list.append(np.array(geom.coords))
                    elif geom.geom_type == 'Polygon':
                        pts_list.append(np.array(geom.exterior.coords))
                    elif geom.geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            pts_list.append(np.array(line.coords))
                    elif geom.geom_type == 'MultiPolygon':
                        for poly in geom.geoms:
                            pts_list.append(np.array(poly.exterior.coords))
                    
                    for pts in pts_list:
                        if pts.shape[1] == 2:
                            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])
                        map_elements[class_name].append(pts)

            except Exception as e:
                print(f"  - Error reading {table_name} in {location}: {e}")
                continue
    
    cnt_str = ", ".join([f"{k}:{len(v)}" for k,v in map_elements.items()])
    print(f"[{location}] 추출 완료. ({cnt_str})")
    return map_elements

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

    unique_locations = list(set([s['map_location'] for s in all_samples if s['map_location']]))
    print(f"Maps to extract: {unique_locations}")

    id2map = {}
    maps_root = os.environ.get("NUPLAN_MAPS_ROOT")
    
    for location in unique_locations:
        map_vectors = get_map_vectors_from_city(maps_root, None, location)
        if map_vectors:
            id2map[location] = map_vectors

    infos = dict(samples=all_samples, id2map=id2map)
    out_file = osp.join(out_dir, f"navsim_map_infos_{split}.pkl")
    mmcv.dump(infos, out_file)
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--nproc', type=int, default=16)
    args = parser.parse_args()

    if os.environ.get("NUPLAN_MAPS_ROOT") is None:
         os.environ['NUPLAN_MAPS_ROOT'] = osp.join(args.data_root, 'download', 'maps')
    
    os.makedirs(args.out_dir, exist_ok=True)

    # for split in ['mini', 'trainval', 'test']:
    for split in ['test']:
        create_navsim_infos(args.data_root, args.out_dir, split, args.nproc)