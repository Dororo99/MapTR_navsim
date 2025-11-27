
import pickle
import sys
import os
from projects.mmdet3d_plugin.datasets.navsim_map_dataset import CustomNavsimLocalMapDataset

# Mock config arguments
data_root = 'data/navsim/'
ann_file = 'data/navsim/navsim_map_infos_test.pkl'
map_classes = ['divider', 'ped_crossing', 'boundary']
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

try:
    print("Initializing dataset...")
    dataset = CustomNavsimLocalMapDataset(
        data_root=data_root,
        ann_file=ann_file,
        map_classes=map_classes,
        pc_range=pc_range,
        pipeline=[],
        classes=['car'],
        test_mode=False
    )
    print("Dataset initialized.")
    
    print("Attempting to pickle dataset...")
    pickle.dumps(dataset)
    print("Pickle successful.")

except TypeError as e:
    print(f"Pickle failed: {e}")
    # Try to find the culprit
    if "dict_keys" in str(e):
        print("Searching for dict_keys in dataset attributes...")
        for k, v in dataset.__dict__.items():
            try:
                pickle.dumps(v)
            except TypeError as sub_e:
                if "dict_keys" in str(sub_e):
                    print(f"Found unpicklable attribute: {k} (Type: {type(v)})")
                    if hasattr(v, '__dict__'):
                        for sub_k, sub_v in v.__dict__.items():
                            try:
                                pickle.dumps(sub_v)
                            except TypeError as sub_sub_e:
                                if "dict_keys" in str(sub_sub_e):
                                    print(f"  -> Found unpicklable sub-attribute: {sub_k} (Type: {type(sub_v)})")

except Exception as e:
    print(f"An error occurred: {e}")
