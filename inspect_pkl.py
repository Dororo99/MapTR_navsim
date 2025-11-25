import mmcv
import os

pkl_path = 'data/navsim/navsim_map_infos_test.pkl'
data = mmcv.load(pkl_path)
print(f"Total samples: {len(data['samples'])}")
if len(data['samples']) > 0:
    sample = data['samples'][0]
    print("Sample 0 keys:", sample.keys())
    if 'cams' in sample:
        for cam_name, cam_info in sample['cams'].items():
            print(f"Cam {cam_name} path: {cam_info['data_path']}")
            break
