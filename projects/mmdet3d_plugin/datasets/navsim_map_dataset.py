import os
import copy
import random
import numpy as np
import mmcv
import torch
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.parallel import DataContainer as DC
from shapely.geometry import LineString, box, MultiLineString, MultiPolygon
from shapely import affinity
import shapely.ops as ops
from mmdet.datasets.pipelines import to_tensor
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from .nuscenes_dataset import CustomNuScenesDataset

# NuPlan Devkit Imports
try:
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
    from nuplan.database.maps_db.map_api import NuPlanMapWrapper
    from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer
except ImportError:
    print("Warning: nuplan-devkit not found.")

class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates (MapTR Standard)"""
    def __init__(self, instance_line_list, sample_dist=1, num_samples=250, padding=False, fixed_num=-1, padding_value=-10000, patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value
        self.instance_list = instance_line_list

    @property
    def fixed_num_sampled_points(self):
        if len(self.instance_list) == 0:
            return torch.zeros((0, self.fixed_num, 2), dtype=torch.float32)
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array).to(dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
                            dtype=torch.float32)
        instance_bbox_tensor[:,0] = torch.clamp(instance_bbox_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,1] = torch.clamp(instance_bbox_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_bbox_tensor[:,2] = torch.clamp(instance_bbox_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,3] = torch.clamp(instance_bbox_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_bbox_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        fixed_num_sampled_points = self.fixed_num_sampled_points
        if len(fixed_num_sampled_points) == 0:
            return torch.zeros((0, 0, self.fixed_num, 2), dtype=torch.float32)
            
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        fixed_num_sampled_points = self.fixed_num_sampled_points
        if len(fixed_num_sampled_points) == 0:
            return torch.zeros((0, 0, self.fixed_num, 2), dtype=torch.float32)
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor
    
    @property
    def shift_fixed_num_sampled_points_v2(self):
        if len(self.instance_list) == 0:
            final_shift_num = self.fixed_num - 1
            return torch.zeros((0, final_shift_num, self.fixed_num, 2), dtype=torch.float32)

        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        if len(self.instance_list) == 0:
            final_shift_num = self.fixed_num - 1
            return torch.zeros((0, final_shift_num * 2, self.fixed_num, 2), dtype=torch.float32)

        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape
            if shifts_num > 2*final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index+shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts,flip1_shifts_pts),axis=0)
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full([final_shift_num*2-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        fixed_num_sampled_points = self.fixed_num_sampled_points
        if len(fixed_num_sampled_points) == 0:
             return torch.zeros((0, 0, self.fixed_num, 2), dtype=torch.float32)

        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num*2, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num*2-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor


class VectorizedLocalMap(object):
    """NuPlan 맵 레이어를 사용한 Map Vector 생성
    
    NuScenes 스타일 매핑:
    - divider (0): lanes_polygons 인접 경계 (차선 사이 구분선만, 중복 제거)
    - ped_crossing (1): crosswalks 레이어
    - boundary (2): road_segments union 외곽선 (도로 가장 바깥 경계만)
    """
    
    CLASS2LABEL = {
        'divider': 0,
        'ped_crossing': 1,
        'crosswalks': 1,
        'boundary': 2,
    }
    
    def __init__(self, dataroot, patch_size, map_classes, sample_dist=1, num_samples=250, padding=False, fixed_ptsnum_per_line=-1, padding_value=-10000, map_root=None):
        self.data_root = dataroot
        if map_root:
            self.map_root = map_root
        else:
            self.map_root = os.path.join(dataroot, 'maps')
        self.map_version = "nuplan-maps-v1.0"
        self.vec_classes = map_classes
        self.patch_size = patch_size
        self.fixed_num = fixed_ptsnum_per_line
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.padding_value = padding_value

        # NuPlan Map DB 초기화
        self.maps_db = GPKGMapsDB(map_root=self.map_root, map_version=self.map_version)
        self.map_apis = {}
        self.map_explorers = {}
        
        self.MAPS = ['us-nv-las-vegas-strip', 'us-ma-boston', 'us-pa-pittsburgh-hazelwood', 'sg-one-north']
        for loc in self.MAPS:
            try:
                self.map_apis[loc] = NuPlanMapWrapper(self.maps_db, loc)
                self.map_explorers[loc] = NuPlanMapExplorer(self.map_apis[loc])
            except Exception as e:
                print(f"Warning: Failed to load map {loc}: {e}")

        self.ped_crossing_layer = 'crosswalks'

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                # lanes_polygons에서 인접 차선 사이 경계만 추출 (중복 제거)
                divider_lines = self.get_lane_dividers(patch_box, patch_angle, location)
                for inst in divider_lines:
                    vectors.append((inst, 0))
                        
            elif vec_class == 'ped_crossing':
                ped_lines = self.get_ped_crossing_lines(patch_box, patch_angle, location)
                for inst in ped_lines:
                    vectors.append((inst, 1))
                    
            elif vec_class == 'boundary':
                # road_segments union 후 가장 바깥 외곽선만
                boundary_lines = self.get_road_boundary_lines(patch_box, patch_angle, location)
                for inst in boundary_lines:
                    vectors.append((inst, 2))

        # LiDARInstanceLines 객체로 변환
        gt_instance = []
        gt_labels = []
        for instance, label in vectors:
            if instance is not None and not instance.is_empty and label != -1:
                gt_instance.append(instance)
                gt_labels.append(label)

        gt_instance_lines = LiDARInstanceLines(
            gt_instance, self.sample_dist, self.num_samples, self.padding, 
            self.fixed_num, self.padding_value, patch_size=self.patch_size
        )

        return dict(gt_vecs_pts_loc=gt_instance_lines, gt_vecs_label=gt_labels)

    def get_lane_dividers(self, patch_box, patch_angle, location):
        """lanes_polygons에서 인접 차선 사이 경계만 추출 (중복 제거)"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        # 1. 모든 lane polygon 수집
        lane_polygons = []
        try:
            records = map_api.load_vector_layer('lanes_polygons')
            for idx, row in records.iterrows():
                geometry = row['geometry']
                if geometry is None or geometry.is_empty:
                    continue
                clipped = geometry.intersection(patch)
                if clipped.is_empty:
                    continue
                clipped = affinity.rotate(clipped, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                clipped = affinity.affine_transform(clipped, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if clipped.geom_type == 'Polygon':
                    lane_polygons.append(clipped)
                elif clipped.geom_type == 'MultiPolygon':
                    lane_polygons.extend(list(clipped.geoms))
        except Exception as e:
            return []
        
        if len(lane_polygons) < 2:
            return []
        
        # 2. 모든 lane polygon을 union
        try:
            all_lanes_union = ops.unary_union(lane_polygons)
        except:
            return []
        
        # 3. 개별 lane polygon들의 경계선 수집
        all_boundaries = []
        for poly in lane_polygons:
            all_boundaries.append(poly.exterior)
        
        # 4. 모든 경계선을 합침
        try:
            all_lines = ops.unary_union(all_boundaries)
        except:
            return []
        
        # 5. Union된 도로 영역의 외곽선 (boundary용)
        if all_lanes_union.geom_type == 'Polygon':
            outer_boundary = all_lanes_union.exterior
        elif all_lanes_union.geom_type == 'MultiPolygon':
            outer_boundary = ops.unary_union([p.exterior for p in all_lanes_union.geoms])
        else:
            outer_boundary = None
        
        # 6. Divider = 전체 경계 - 외곽 경계 (차선 사이 경계만 남김)
        if outer_boundary is not None:
            try:
                divider_lines = all_lines.difference(outer_boundary.buffer(0.5))
            except:
                divider_lines = all_lines
        else:
            divider_lines = all_lines
        
        # 7. LineString으로 변환
        line_list = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        def extract_lines(geom):
            if geom is None or geom.is_empty:
                return
            if geom.geom_type == 'LineString':
                clipped = geom.intersection(local_patch)
                if not clipped.is_empty and clipped.length > 2.0:
                    if clipped.geom_type == 'LineString':
                        line_list.append(clipped)
                    elif clipped.geom_type == 'MultiLineString':
                        for line in clipped.geoms:
                            if line.length > 2.0:
                                line_list.append(line)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    extract_lines(line)
            elif geom.geom_type == 'GeometryCollection':
                for g in geom.geoms:
                    extract_lines(g)
            elif geom.geom_type == 'LinearRing':
                extract_lines(LineString(geom.coords))
        
        extract_lines(divider_lines)
        return line_list

    def get_road_boundary_lines(self, patch_box, patch_angle, location):
        """road_segments를 union해서 도로 가장 바깥 외곽선만 추출"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        try:
            records = map_api.load_vector_layer('road_segments')
            for idx, row in records.iterrows():
                geometry = row['geometry']
                if geometry is None or geometry.is_empty:
                    continue
                new_polygon = geometry.intersection(patch)
                if new_polygon.is_empty:
                    continue
                new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_polygon.geom_type == 'Polygon':
                    polygon_list.append(new_polygon)
                elif new_polygon.geom_type == 'MultiPolygon':
                    polygon_list.extend(list(new_polygon.geoms))
        except Exception as e:
            return []
        
        if not polygon_list:
            return []
        
        # 모든 road_segments를 union → 전체 도로 영역
        try:
            union_road = ops.unary_union(polygon_list)
        except:
            return []
        
        if union_road.is_empty:
            return []
        
        # 외곽선만 추출
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        if union_road.geom_type == 'Polygon':
            union_road = MultiPolygon([union_road])
        
        lines = []
        for poly in union_road.geoms:
            # 외곽선 (도로 가장자리)
            ext = LineString(poly.exterior.coords)
            clipped = ext.intersection(local_patch)
            if not clipped.is_empty:
                if clipped.geom_type == 'MultiLineString':
                    clipped = ops.linemerge(clipped)
                if clipped.geom_type == 'LineString' and clipped.length > 2.0:
                    lines.append(clipped)
                elif clipped.geom_type == 'MultiLineString':
                    for line in clipped.geoms:
                        if line.length > 2.0:
                            lines.append(line)
            
            # 내부 홀 (섬 등)
            for interior in poly.interiors:
                inter_line = LineString(interior.coords)
                clipped = inter_line.intersection(local_patch)
                if not clipped.is_empty:
                    if clipped.geom_type == 'MultiLineString':
                        clipped = ops.linemerge(clipped)
                    if clipped.geom_type == 'LineString' and clipped.length > 2.0:
                        lines.append(clipped)
                    elif clipped.geom_type == 'MultiLineString':
                        for line in clipped.geoms:
                            if line.length > 2.0:
                                lines.append(line)
        
        return lines

    def get_boundary_lines(self, patch_box, patch_angle, location, boundary_types):
        """boundaries 레이어에서 특정 type의 경계선 추출 (Legacy)"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        line_list = []
        try:
            records = map_api.load_vector_layer('boundaries')
            for idx, row in records.iterrows():
                # boundary_type_fid 필터링
                if row['boundary_type_fid'] not in boundary_types:
                    continue
                
                geometry = row['geometry']
                if geometry is None or geometry.is_empty:
                    continue
                new_line = geometry.intersection(patch)
                if new_line.is_empty:
                    continue
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line.geoms:
                        if single_line.length > 1.0:
                            line_list.append(single_line)
                elif new_line.geom_type == 'LineString':
                    if new_line.length > 1.0:
                        line_list.append(new_line)
                elif new_line.geom_type == 'GeometryCollection':
                    for g in new_line.geoms:
                        if g.geom_type == 'LineString' and g.length > 1.0:
                            line_list.append(g)
        except Exception as e:
            pass
        
        return line_list

    def get_divider_lines(self, patch_box, patch_angle, location):
        """Legacy: baseline_paths에서 차선 중심선 추출 (더 이상 사용 안함)"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        line_list = []
        try:
            records = map_api.load_vector_layer('baseline_paths')
            for geometry in records['geometry']:
                if geometry is None or geometry.is_empty:
                    continue
                new_line = geometry.intersection(patch)
                if new_line.is_empty:
                    continue
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line.geoms:
                        if single_line.length > 1.0:
                            line_list.append(single_line)
                elif new_line.geom_type == 'LineString':
                    if new_line.length > 1.0:
                        line_list.append(new_line)
                elif new_line.geom_type == 'GeometryCollection':
                    for g in new_line.geoms:
                        if g.geom_type == 'LineString' and g.length > 1.0:
                            line_list.append(g)
        except Exception as e:
            pass
        
        return line_list

    def get_ped_crossing_lines(self, patch_box, patch_angle, location):
        """crosswalks에서 횡단보도 외곽선 추출"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        try:
            records = map_api.load_vector_layer(self.ped_crossing_layer)
            for geometry in records['geometry']:
                if geometry is None or geometry.is_empty:
                    continue
                new_polygon = geometry.intersection(patch)
                if new_polygon.is_empty:
                    continue
                new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_polygon.geom_type == 'Polygon':
                    polygon_list.append(new_polygon)
                elif new_polygon.geom_type == 'MultiPolygon':
                    polygon_list.extend(list(new_polygon.geoms))
        except Exception as e:
            pass
        
        if not polygon_list:
            return []
        
        # 각 polygon의 exterior를 LineString으로 변환
        lines = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        for poly in polygon_list:
            ext = LineString(poly.exterior.coords)
            clipped = ext.intersection(local_patch)
            if clipped.is_empty:
                continue
            if clipped.geom_type == 'MultiLineString':
                clipped = ops.linemerge(clipped)
            if clipped.geom_type == 'LineString' and clipped.length > 1.0:
                lines.append(clipped)
            elif clipped.geom_type == 'MultiLineString':
                for line in clipped.geoms:
                    if line.length > 1.0:
                        lines.append(line)
        
        return lines

    def get_road_boundary_lines(self, patch_box, patch_angle, location):
        """road_segments를 union해서 도로 외곽선만 추출"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        try:
            records = map_api.load_vector_layer(self.boundary_layer)
            for geometry in records['geometry']:
                if geometry is None or geometry.is_empty:
                    continue
                new_polygon = geometry.intersection(patch)
                if new_polygon.is_empty:
                    continue
                new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_polygon.geom_type == 'Polygon':
                    polygon_list.append(new_polygon)
                elif new_polygon.geom_type == 'MultiPolygon':
                    polygon_list.extend(list(new_polygon.geoms))
        except Exception as e:
            pass
        
        if not polygon_list:
            return []
        
        # 모든 road_segments를 union → 전체 도로 영역
        try:
            union_road = ops.unary_union(polygon_list)
        except:
            return []
        
        if union_road.is_empty:
            return []
        
        # 외곽선만 추출
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        if union_road.geom_type == 'Polygon':
            union_road = MultiPolygon([union_road])
        
        lines = []
        for poly in union_road.geoms:
            # 외곽선 (도로 가장자리)
            ext = LineString(poly.exterior.coords)
            clipped = ext.intersection(local_patch)
            if not clipped.is_empty:
                if clipped.geom_type == 'MultiLineString':
                    clipped = ops.linemerge(clipped)
                if clipped.geom_type == 'LineString' and clipped.length > 1.0:
                    lines.append(clipped)
                elif clipped.geom_type == 'MultiLineString':
                    for line in clipped.geoms:
                        if line.length > 1.0:
                            lines.append(line)
            
            # 내부 홀 (섬 등)
            for interior in poly.interiors:
                inter_line = LineString(interior.coords)
                clipped = inter_line.intersection(local_patch)
                if not clipped.is_empty:
                    if clipped.geom_type == 'MultiLineString':
                        clipped = ops.linemerge(clipped)
                    if clipped.geom_type == 'LineString' and clipped.length > 1.0:
                        lines.append(clipped)
                    elif clipped.geom_type == 'MultiLineString':
                        for line in clipped.geoms:
                            if line.length > 1.0:
                                lines.append(line)
        
        return lines

    # 아래 메서드들은 더 이상 사용되지 않음 - 호환성을 위해 유지
    def get_map_geom(self, patch_box, patch_angle, layer_names, location, geom_type='line'):
        """레이어에서 기하 정보 추출 (Legacy)"""
        map_geom = []
        for layer_name in layer_names:
            if geom_type == 'line':
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
            else:  # polygon
                geoms = self.get_polygon(patch_box, patch_angle, layer_name, location)
            map_geom.append((layer_name, geoms))
        return map_geom

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        """Line 레이어 추출 (Legacy)"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        line_list = []
        try:
            records = map_api.load_vector_layer(layer_name)
            for geometry in records['geometry']:
                if geometry is None or geometry.is_empty:
                    continue
                new_line = geometry.intersection(patch)
                if new_line.is_empty:
                    continue
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_line.geom_type == 'MultiLineString':
                    line_list.extend(list(new_line.geoms))
                elif new_line.geom_type == 'LineString':
                    line_list.append(new_line)
                elif new_line.geom_type == 'GeometryCollection':
                    for g in new_line.geoms:
                        if g.geom_type == 'LineString':
                            line_list.append(g)
        except Exception as e:
            pass
        
        return line_list

    def get_polygon(self, patch_box, patch_angle, layer_name, location):
        """Polygon 레이어 (crosswalks, lanes_polygons) 추출"""
        if location not in self.map_apis:
            return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        try:
            records = map_api.load_vector_layer(layer_name)
            for geometry in records['geometry']:
                if geometry is None or geometry.is_empty:
                    continue
                new_polygon = geometry.intersection(patch)
                if new_polygon.is_empty:
                    continue
                new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                
                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)
        except Exception as e:
            pass
        
        return polygon_list

    def _one_type_line_geom_to_instances(self, line_geom):
        """Line geometry를 LineString 인스턴스 리스트로 변환"""
        line_instances = []
        for line in line_geom:
            if line is None or line.is_empty:
                continue
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    if single_line.length > 1.0:  # 최소 길이 필터
                        line_instances.append(single_line)
            elif line.geom_type == 'LineString':
                if line.length > 1.0:
                    line_instances.append(line)
        return line_instances

    def line_geoms_to_instances(self, line_geom):
        """Line geometry dict를 인스턴스 dict로 변환"""
        line_instances_dict = {}
        for line_type, lines in line_geom:
            instances = self._one_type_line_geom_to_instances(lines)
            line_instances_dict[line_type] = instances
        return line_instances_dict

    def ped_poly_geoms_to_instances(self, ped_geom):
        """Pedestrian crossing polygon → 외곽선 LineString 추출"""
        if not ped_geom or len(ped_geom) == 0:
            return []
        
        ped_polygons = ped_geom[0][1] if len(ped_geom) > 0 else []
        if len(ped_polygons) == 0:
            return []
        
        # Union all polygons
        all_polys = []
        for mp in ped_polygons:
            if mp is None or mp.is_empty:
                continue
            if mp.geom_type == 'MultiPolygon':
                all_polys.extend(list(mp.geoms))
            elif mp.geom_type == 'Polygon':
                all_polys.append(mp)
        
        if not all_polys:
            return []
        
        try:
            union_segments = ops.unary_union(all_polys)
        except:
            return []
        
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        
        results = []
        for poly in union_segments.geoms:
            ext = poly.exterior
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        """Lane polygon들을 union하고 외곽선(contour) 추출 → boundary"""
        if not polygon_geom:
            return []
        
        # Collect all polygons
        all_polys = []
        for layer_name, geoms in polygon_geom:
            for mp in geoms:
                if mp is None or mp.is_empty:
                    continue
                if mp.geom_type == 'MultiPolygon':
                    all_polys.extend(list(mp.geoms))
                elif mp.geom_type == 'Polygon':
                    all_polys.append(mp)
        
        if not all_polys:
            return []
        
        # Union all lane polygons → 전체 도로 영역
        try:
            union_segments = ops.unary_union(all_polys)
        except:
            return []
        
        if union_segments.is_empty:
            return []
        
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        
        exteriors = []
        interiors = []
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)
        
        results = []
        # Exterior boundaries (도로 외곽)
        for ext in exteriors:
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        # Interior boundaries (도로 내 홀, 예: 섬)
        for inter in interiors:
            if not inter.is_ccw:
                inter = LineString(list(inter.coords)[::-1])
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        return self._one_type_line_geom_to_instances(results)

@DATASETS.register_module()
class CustomNavsimLocalMapDataset(CustomNuScenesDataset):
    MAPCLASSES = ('divider', 'ped_crossing', 'boundary')

    def __init__(self, map_ann_file=None, map_fixed_ptsnum_per_line=-1, sensor_root=None, **kwargs):
        self.pc_range = kwargs.pop('pc_range', None)
        self.map_classes = kwargs.pop('map_classes', None)
        self.padding_value = kwargs.pop('padding_value', -10000)
        self.eval_use_same_gt_sample_num_flag = kwargs.pop('eval_use_same_gt_sample_num_flag', False)
        
        if 'fixed_ptsnum_per_line' in kwargs:
            self.fixed_num = kwargs.pop('fixed_ptsnum_per_line')
        else:
            self.fixed_num = map_fixed_ptsnum_per_line

        super().__init__(**kwargs)
        self.map_ann_file = map_ann_file
        self.sensor_root = sensor_root # 이미지 절대 경로 구성을 위해 필요

        # BEV Patch Size 설정
        if self.pc_range is not None:
            patch_h = self.pc_range[4] - self.pc_range[1]
            patch_w = self.pc_range[3] - self.pc_range[0]
            self.patch_size = (patch_h, patch_w)

        # Check if using pre-generated maps by loading first sample
        use_pregenerated = False
        if hasattr(self, 'data_infos') and len(self.data_infos) > 0:
            first_sample = self.data_infos[0]
            if 'map_available' in first_sample and first_sample['map_available']:
                use_pregenerated = True
                print(f"Using pre-generated maps - skipping VectorizedLocalMap initialization")
        
        # Only initialize VectorizedLocalMap if NOT using pre-generated maps
        if not use_pregenerated:
            print(f"Initializing VectorizedLocalMap for runtime map generation...")
            data_root = kwargs.get('data_root')
            map_root = os.path.join(data_root, 'download', 'maps')
            if not os.path.exists(map_root):
                map_root = os.path.join(data_root, 'maps')

            self.vector_map = VectorizedLocalMap(
                dataroot=data_root,
                patch_size=self.patch_size,
                map_classes=self.MAPCLASSES,
                fixed_ptsnum_per_line=self.fixed_num,
                map_root=map_root
            )
        else:
            self.vector_map = None  # Not needed with pre-generated maps

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations.
        """
        data = mmcv.load(ann_file)
        if 'samples' in data:
            data_infos = data['samples']
        elif 'infos' in data:
            data_infos = data['infos']
        else:
            raise KeyError(f"Annotation file {ann_file} must contain 'samples' or 'infos' key.")
            
        # Sort by timestamp if available
        if len(data_infos) > 0 and 'timestamp' in data_infos[0]:
            data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
            
        # Set version if not present (NuScenesDataset expects it)
        if not hasattr(self, 'version'):
             self.version = 'v1.0-trainval' # Default or dummy
             
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info.get('lidar_path', ''),
            timestamp=info['timestamp'] / 1e6,
            map_location=info['map_location'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info.get('lidar2ego_translation', np.zeros(3)),
            lidar2ego_rotation=info.get('lidar2ego_rotation', np.array([1,0,0,0])),
            # Temporal fields for queue
            prev_idx=info.get('prev', ''),
            next_idx=info.get('next', ''),
            scene_token=info.get('scene_token', ''),
            frame_idx=info.get('frame_idx', 0),
            sweeps=info.get('sweeps', []),
        )
        
        # Camera handling - NuScenes style
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            for cam_name, cam_info in info['cams'].items():
                img_path = cam_info['data_path']
                if self.sensor_root:
                    if img_path.startswith('data/navsim/download/'):
                        img_path = img_path.replace('data/navsim/download/', '')
                        img_path = os.path.join(self.sensor_root, img_path)
                    elif not img_path.startswith('/'):
                        img_path = os.path.join(self.sensor_root, img_path)
                image_paths.append(img_path)
                
                # Compute lidar2cam from sensor2lidar (NuScenes way)
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            
            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))
        
        # Load can_bus from info first
        can_bus = info.get('can_bus', np.zeros(18))
        
        # CAN Bus processing (same as NuScenes)
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus[:3] = translation
        can_bus[3:7] = rotation.elements # Use the elements of the Quaternion object
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus
        
        # Add pre-generated map data if available
        if 'map_available' in info and info['map_available']:
            input_dict['map_available'] = True
            input_dict['map_gt_labels'] = info['map_gt_labels']
            input_dict['map_gt_pts_loc'] = info['map_gt_pts_loc']
            input_dict['map_gt_pts_loc_shift'] = info.get('map_gt_pts_loc_shift', None)
        
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        # Check if map data was pre-generated and stored in PKL
        if 'map_available' in input_dict and input_dict['map_available']:
            # Use pre-generated map data (much faster!)
            gt_vecs_label = to_tensor(input_dict['map_gt_labels'])
            
            # Reconstruct LiDARInstanceLines from stored data
            fixed_pts = input_dict['map_gt_pts_loc']
            
            # Create dummy LiDARInstanceLines for compatibility
            instance_lines = LiDARInstanceLines(
                [], 1, 250, False, self.fixed_num, self.padding_value, 
                patch_size=self.patch_size
            )
            # Override with pre-generated data
            instance_lines._fixed_pts = torch.from_numpy(fixed_pts).float()
            
            gt_vecs_pts_loc = instance_lines
        else:
            # Fall back to runtime generation (slower)
            location = input_dict['map_location']
            e2g_t = input_dict['ego2global_translation']
            e2g_r = input_dict['ego2global_rotation']
            l2e_t = input_dict['lidar2ego_translation']
            l2e_r = input_dict['lidar2ego_rotation']

            # Compute Lidar2Global
            T_global_ego = np.eye(4)
            T_global_ego[:3, :3] = Quaternion(e2g_r).rotation_matrix
            T_global_ego[:3, 3] = e2g_t

            T_ego_lidar = np.eye(4)
            T_ego_lidar[:3, :3] = Quaternion(l2e_r).rotation_matrix
            T_ego_lidar[:3, 3] = l2e_t

            T_global_lidar = T_global_ego @ T_ego_lidar
            
            l2g_t = T_global_lidar[:3, 3]
            l2g_r = Quaternion(matrix=T_global_lidar[:3, :3]).elements

            anns_results = self.vector_map.gen_vectorized_samples(location, l2g_t, l2g_r)
            
            gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        return example
    
    
    def prepare_train_data(self, index):
        """
        Training data preparation with temporal queue.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []

        # NuScenes pattern: shuffle prev frames, sort, then add current
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            example = self.vectormap_pipeline(example, input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue