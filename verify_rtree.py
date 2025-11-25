
import sys
import os
import numpy as np
from nuscenes.eval.common.utils import Quaternion

# Add project root to path
sys.path.append(os.getcwd())

from projects.mmdet3d_plugin.datasets.navsim_map_dataset import VectorizedNavsimMap

def test_rtree():
    print("Initializing VectorizedNavsimMap...")
    # Mock data root and patch size
    dataroot = 'data/navsim'
    patch_size = (102.4, 102.4)
    
    vec_map = VectorizedNavsimMap(
        dataroot=dataroot,
        patch_size=patch_size
    )
    
    # Mock map elements
    print("Creating mock map elements...")
    map_elements = {
        'divider': [np.random.rand(10, 3) * 100 for _ in range(1000)],
        'ped_crossing': [np.random.rand(10, 3) * 100 for _ in range(100)],
        'boundary': [np.random.rand(10, 3) * 100 for _ in range(1000)]
    }
    
    location = 'mock_location'
    ego2global_translation = np.array([50, 50, 0])
    ego2global_rotation = np.array([1, 0, 0, 0]) # Identity quaternion
    
    print("Generating vectorized samples (First run - should build index)...")
    import time
    start = time.time()
    vec_map.gen_vectorized_samples(location, map_elements, ego2global_translation, ego2global_rotation)
    end = time.time()
    print(f"First run took: {end - start:.4f} seconds")
    
    print("Generating vectorized samples (Second run - should use cached index)...")
    start = time.time()
    vec_map.gen_vectorized_samples(location, map_elements, ego2global_translation, ego2global_rotation)
    end = time.time()
    print(f"Second run took: {end - start:.4f} seconds")
    
    print("Verification successful!")

if __name__ == "__main__":
    test_rtree()
