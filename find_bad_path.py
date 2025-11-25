import mmcv
import os

pkl_path = 'data/navsim/navsim_map_infos_test.pkl'
print(f"Loading {pkl_path}...")
data = mmcv.load(pkl_path)
print(f"Loaded {len(data['samples'])} samples")

target_file = 'c74afe5f0f2551a0.jpg'
found = False
for i, sample in enumerate(data['samples']):
    if 'cams' in sample:
        for cam_name, cam_info in sample['cams'].items():
            if target_file in cam_info['data_path']:
                print(f"Found target file in sample {i}")
                print(f"Path in pkl: {cam_info['data_path']}")
                found = True
                break
    if found: break

if not found:
    print("Target file not found in pkl")
