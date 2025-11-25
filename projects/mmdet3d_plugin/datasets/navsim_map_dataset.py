import copy
import numpy as np
from mmdet.datasets import DATASETS
import mmcv
import torch
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from shapely.geometry import LineString, box, MultiLineString
from mmdet.datasets.pipelines import to_tensor

class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates
    """
    def __init__(self, 
                 instance_line_list, 
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
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
    def start_end_points(self):
        if len(self.instance_list) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
                                dtype=torch.float32)
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        if len(self.instance_list) == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        instance_bbox_list = []
        for instance in self.instance_list:
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
    def fixed_num_sampled_points(self):
        if len(self.instance_list) == 0:
            return torch.zeros((0, self.fixed_num, 2), dtype=torch.float32)
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        fixed_num_sampled_points = self.fixed_num_sampled_points
        if len(fixed_num_sampled_points) == 0:
            return torch.zeros((0, 0, self.fixed_num, 2), dtype=torch.float32) # Shape might need adjustment based on usage
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
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
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
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        if len(self.instance_list) == 0:
            # Return empty tensor with shape (0, num_shifts, fixed_num, 2)
            # We need to determine num_shifts. For v2, it seems variable or fixed?
            # In v2 logic: final_shift_num = self.fixed_num - 1
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
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
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
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full([final_shift_num*2-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        fixed_num_sampled_points = self.fixed_num_sampled_points
        if len(fixed_num_sampled_points) == 0:
            # v4 logic seems to produce variable shift num?
            # It pads to shift_num*2 - ... wait.
            # Let's assume it returns empty tensor if input is empty.
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
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

class VectorizedNavsimMap(object):
    CLASS2LABEL = {
        'divider': 0,
        'ped_crossing': 1,
        'boundary': 2,
        'stop_line': 3, 
        'centerline': 4,
        'others': -1
    }
    def __init__(self,
                 dataroot,
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000):
        self.data_root = dataroot
        self.vec_classes = map_classes
        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value
        
        # Spatial Index Cache
        self.map_indices = {} # location -> (rtree_index, flattened_elements)

    def _get_map_index(self, location, map_elements):
        if location in self.map_indices:
            return self.map_indices[location]
        
        from rtree import index
        idx = index.Index()
        flattened_elements = []
        cursor = 0
        
        print(f"Building R-tree index for {location}...")
        
        for vec_class in self.vec_classes:
            if vec_class not in map_elements:
                continue
            
            geoms = map_elements[vec_class]
            for geom in geoms:
                # geom is (N, 3) array
                min_x, min_y = np.min(geom[:, :2], axis=0)
                max_x, max_y = np.max(geom[:, :2], axis=0)
                
                idx.insert(cursor, (min_x, min_y, max_x, max_y))
                flattened_elements.append((vec_class, geom))
                cursor += 1
                
        self.map_indices[location] = (idx, flattened_elements)
        return idx, flattened_elements

    def gen_vectorized_samples(self, location, map_elements, ego2global_translation, ego2global_rotation):
        # Ego 2 Global 변환 행렬 생성
        # R: Quaternion to Matrix, T: translation
        R = Quaternion(ego2global_rotation).rotation_matrix
        T = ego2global_translation
        
        # Get or build spatial index
        rtree_idx, all_elements = self._get_map_index(location, map_elements)

        # Calculate Global Patch Bounds for Query
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        
        # Corners in Ego frame
        corners_ego = np.array([
            [max_x, max_y, 0],
            [max_x, -max_y, 0],
            [-max_x, -max_y, 0],
            [-max_x, max_y, 0]
        ])
        
        # Transform to Global frame: P_global = R * P_ego + T
        corners_global = (corners_ego @ R.T) + T
        
        min_gx = np.min(corners_global[:, 0])
        min_gy = np.min(corners_global[:, 1])
        max_gx = np.max(corners_global[:, 0])
        max_gy = np.max(corners_global[:, 1])
        
        # Query R-tree
        candidate_indices = list(rtree_idx.intersection((min_gx, min_gy, max_gx, max_gy)))
        
        if len(candidate_indices) == 0:
            print(f"DEBUG: R-tree query failed for {location}")
            print(f"  Ego Translation: {T}")
            print(f"  Global Bounds: ({min_gx}, {min_gy}) - ({max_gx}, {max_gy})")
            # Check first element in map to see where it is
            if len(all_elements) > 0:
                _, first_geom = all_elements[0]
                f_min_x, f_min_y = np.min(first_geom[:, :2], axis=0)
                f_max_x, f_max_y = np.max(first_geom[:, :2], axis=0)
                print(f"  Sample Map Element Bounds: ({f_min_x}, {f_min_y}) - ({f_max_x}, {f_max_y})")
            else:
                print("  Map elements are empty!")
        
        vectors = []
        local_patch = box(-max_x, -max_y, max_x, max_y)

        for i in candidate_indices:
            vec_class, geom_global = all_elements[i]
            
            # Global -> Ego 변환
            pts_global = geom_global[:, :3]
            pts_ego = (pts_global - T) @ R
            
            # Shapely LineString 생성 (2D, x/y)
            try:
                line_ego = LineString(pts_ego[:, :2])
            except:
                continue
            
            if line_ego.is_empty:
                continue

            # Patch 안에 들어오는지 확인 (Intersection)
            if not line_ego.intersects(local_patch):
                continue
            
            line_ego_clipped = line_ego.intersection(local_patch)

            if line_ego_clipped.is_empty:
                continue
            
            # MultiLineString이면 분리해서 추가
            if line_ego_clipped.geom_type == 'MultiLineString':
                for single_line in line_ego_clipped.geoms:
                    if single_line.length < 0.1: continue
                    vectors.append((single_line, self.CLASS2LABEL.get(vec_class, -1)))
            elif line_ego_clipped.geom_type == 'LineString':
                if line_ego_clipped.length < 0.1: continue
                vectors.append((line_ego_clipped, self.CLASS2LABEL.get(vec_class, -1)))
        
        # 필터링 및 LiDARInstanceLines 객체 생성
        gt_instance = []
        gt_labels = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        
        if len(gt_instance) > 0:
             print(f"DEBUG: Found {len(gt_instance)} vectors for {location}")
        else:
             print(f"DEBUG: No vectors found for {location}")
        
        gt_instance = LiDARInstanceLines(gt_instance, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num, self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
        )
        return anns_results

@DATASETS.register_module()
class CustomNavsimLocalMapDataset(CustomNuScenesDataset):
    MAPCLASSES = ('divider', 'ped_crossing', 'boundary')
    
    def __init__(self,
                 map_ann_file=None,
                 queue_length=4,
                 bev_size=(200, 200),
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False,
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_ann_file = map_ann_file
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag

        self.vector_map = VectorizedNavsimMap(
            kwargs.get('data_root'),
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value
        )
        self.is_vis_on_test = False
    
    @classmethod
    def get_map_classes(cls, map_classes=None):
        if map_classes is None:
            return cls.MAPCLASSES
        if isinstance(map_classes, str):
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')
        return class_names

    def load_annotations(self, ann_file):
        print(f"Loading Navsim annotations from {ann_file}")
        data = mmcv.load(ann_file)
        self.id2map = data['id2map']
        data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['sample_idx'],
            timestamp=info['timestamp'],
            map_location=info['map_location'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar_path=info.get('lidar_path', ''),
        )

        # Camera Info Load
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            # info['cams']는 {cam_name: dict} 형태
            for cam_name, cam_info in info['cams'].items():
                data_path = cam_info['data_path']
                # Fix nested directory structure: sensor_blobs/{split}/{split}/...
                # Check for mini, trainval, or test splits
                # for split in ['mini', 'trainval', 'test']:
                #     pattern = f'sensor_blobs/{split}/'
                #     nested_pattern = f'sensor_blobs/{split}/{split}/'
                #     if pattern in data_path and nested_pattern not in data_path:
                #         data_path = data_path.replace(pattern, nested_pattern)
                #         break
                image_paths.append(data_path)
                
                # MapTR 파이프라인을 위한 행렬 준비
                # lidar2cam_rt (Extrinsics)
                l2c_rt = cam_info['lidar2cam_rt']
                
                # Intrinsics
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                
                # lidar2img = intrinsic @ lidar2cam
                l2i_rt = viewpad @ l2c_rt
                
                lidar2img_rts.append(l2i_rt)
                lidar2cam_rts.append(l2c_rt)
                cam_intrinsics.append(viewpad)
            
            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))
        
        # can_bus (dummy if not exists)
        input_dict['can_bus'] = np.zeros(18)
        
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        location = input_dict['map_location']
        e2g_translation = input_dict['ego2global_translation']
        e2g_rotation = input_dict['ego2global_rotation']
        
        if location not in self.id2map:
             raise ValueError(f"Map {location} not found in id2map")

        map_elements = self.id2map[location]
        anns_results = self.vector_map.gen_vectorized_samples(
            location, map_elements, e2g_translation, e2g_rotation
        )
        
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label']).long()
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc'] # LiDARInstanceLines object

        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        
        return example

    def prepare_train_data(self, index):
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            # print(f"Preparing data for index {i}") # Debug logging
            input_dict = self.get_data_info(i)
            if input_dict is None: return None
            
            # Add vectormap GT data to input_dict before pipeline
            location = input_dict['map_location']
            e2g_translation = input_dict['ego2global_translation']
            e2g_rotation = input_dict['ego2global_rotation']
            
            if location not in self.id2map:
                raise ValueError(f"Map {location} not found in id2map")

            map_elements = self.id2map[location]
            anns_results = self.vector_map.gen_vectorized_samples(
                location, map_elements, e2g_translation, e2g_rotation
            )
            
            # Add GT data to input_dict so pipeline can access it
            input_dict['gt_labels_3d'] = to_tensor(anns_results['gt_vecs_label']).long()
            input_dict['gt_bboxes_3d'] = anns_results['gt_vecs_pts_loc']
            
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            
            
            if self.filter_empty_gt and \
                    (example is None or len(example['gt_labels_3d']._data) == 0 or \
                     ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
            
        return self.union2one(queue)

    def union2one(self, queue):
        # CustomNuScenesDataset의 union2one 로직 사용 (이미 상속받음)
        # 하지만 여기서는 간단히 재구현하거나 상속된 것을 사용
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev_exists'] = False
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