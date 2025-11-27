
import mmcv
import sys

file_path = 'data/navsim/navsim_map_infos_test.pkl'
try:
    data = mmcv.load(file_path)
    print(f"Type of data: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'samples' in data and len(data['samples']) > 0:
            sample = data['samples'][0]
            if 'cams' in sample:
                first_cam_key = list(sample['cams'].keys())[0]
                print(f"Sample image path: {sample['cams'][first_cam_key]['data_path']}")
    elif isinstance(data, list):
        print(f"Length of data: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {data[0].keys() if isinstance(data[0], dict) else 'N/A'}")
except Exception as e:
    print(f"Error loading file: {e}")
