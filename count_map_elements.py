
import os
import geopandas as gpd
from pathlib import Path

# Hardcoded path based on user's previous actions
maps_root = "/data3/navsim/download/maps"
location = "us-ma-boston"

map_dir = os.path.join(maps_root, location)
if not os.path.exists(map_dir):
    print(f"Map dir not found: {map_dir}")
    exit()

subdirs = sorted([d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))])
if not subdirs:
    print("No version subdirs found")
    exit()

target_version = subdirs[-1]
gpkg_path = os.path.join(map_dir, target_version, "map.gpkg")
print(f"Reading {gpkg_path}")

layers = ['lanes_polygons', 'lane_connectors', 'crosswalks', 'boundaries', 'generic_drivable_areas']
total_elements = 0

for layer in layers:
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
        count = len(gdf)
        print(f"Layer {layer}: {count} elements")
        total_elements += count
    except Exception as e:
        print(f"Layer {layer}: Error {e}")

print(f"Total elements: {total_elements}")
