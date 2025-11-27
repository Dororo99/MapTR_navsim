import os
import copy
import numpy as np
import mmcv
import torch
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.parallel import DataContainer as DC
from shapely.geometry import LineString, box, MultiLineString, MultiPolygon
from shapely import affinity
from mmdet.datasets.pipelines import to_tensor
from pyquaternion import Quaternion
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
    """VAD의 로직을 MapTR용으로 수정"""
    def __init__(self, dataroot, patch_size, map_classes, sample_dist=1, num_samples=250, padding=False, fixed_ptsnum_per_line=-1, padding_value=-10000, map_root=None):
        self.data_root = dataroot
        if map_root:
            self.map_root = map_root
        else:
            self.map_root = os.path.join(dataroot, 'maps') # maps 폴더 경로 추론
        self.map_version = "nuplan-maps-v1.0"
        self.vec_classes = map_classes
        self.patch_size = patch_size
        self.fixed_num = fixed_ptsnum_per_line
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.padding_value = padding_value

        # NuPlan Map DB 초기화 (메모리에 로드)
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

        # MapTR 클래스 <-> NuPlan 레이어 매핑
        self.layer_mapping = {
            'divider': ['lanes_polygons', 'lane_connectors'], 
            'ped_crossing': ['crosswalks'],
            'boundary': ['boundaries', 'generic_drivable_areas'],
            # 필요시 추가: 'centerline': ['baseline_paths']
        }

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        
        # Patch Box 정의 (x, y, h, w) - VAD 코드 참고
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        
        vectors = []
        for vec_class in self.vec_classes:
            layers = self.layer_mapping.get(vec_class, [])
            for layer_name in layers:
                # 해당 레이어의 기하 정보 쿼리 (VAD 로직 사용)
                geoms = self.get_map_geom(patch_box, patch_angle, layer_name, location)
                for geom in geoms:
                    vectors.append((geom, self.vec_classes.index(vec_class)))

        # LiDARInstanceLines 객체로 변환
        gt_instance = []
        gt_labels = []
        for instance, label in vectors:
            gt_instance.append(instance)
            gt_labels.append(label)

        gt_instance_lines = LiDARInstanceLines(
            gt_instance, self.sample_dist, self.num_samples, self.padding, 
            self.fixed_num, self.padding_value, patch_size=self.patch_size
        )

        return dict(gt_vecs_pts_loc=gt_instance_lines, gt_vecs_label=gt_labels)

    def get_map_geom(self, patch_box, patch_angle, layer_name, location):
        if location not in self.map_apis: return []
        
        map_api = self.map_apis[location]
        patch_x, patch_y = patch_box[0], patch_box[1]
        
        # Patch 좌표 계산
        patch = map_api.get_patch_coord(patch_box, patch_angle)
        
        # 해당 레이어 로드 (GeoDataFrame)
        try:
            records = map_api.load_vector_layer(layer_name)
        except:
            return []

        geom_list = []
        # R-Tree나 인덱스 없이 전체 순회는 느리지만 VAD 방식 따름 (최적화 가능)
        for geometry in records['geometry']:
            if geometry is None or geometry.is_empty: continue
            
            # 1. Intersection Check
            if not geometry.intersects(patch): continue
            new_geom = geometry.intersection(patch)
            if new_geom.is_empty: continue
            
            # 2. Transform to Local Coordinates (Global -> Ego)
            # Rotate (-patch_angle) & Translate (-patch_x, -patch_y)
            new_geom = affinity.rotate(new_geom, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_geom = affinity.affine_transform(new_geom, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            
            # 3. Convert to LineString (MapTR은 선만 처리함)
            if new_geom.geom_type == 'Polygon':
                geom_list.append(new_geom.exterior) # 폴리곤은 외곽선만
            elif new_geom.geom_type == 'MultiPolygon':
                for poly in new_geom.geoms:
                    geom_list.append(poly.exterior)
            elif new_geom.geom_type == 'MultiLineString':
                for line in new_geom.geoms:
                    geom_list.append(line)
            elif new_geom.geom_type == 'LineString':
                geom_list.append(new_geom)
                
        return geom_list

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

        # VAD 스타일의 실시간 맵 로더 초기화
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
            timestamp=info['timestamp'],
            map_location=info['map_location'], # 중요: 맵 쿼리용 키
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        
        # Camera Path Handling (VAD 방식)
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsics = []
            
            for cam_name, cam_info in info['cams'].items():
                # .pkl에는 상대 경로가 있을 수 있으므로 sensor_root와 결합
                img_path = cam_info['data_path']
                if self.sensor_root:
                    if img_path.startswith('data/navsim/download/'):
                        img_path = img_path.replace('data/navsim/download/', '')
                        img_path = os.path.join(self.sensor_root, img_path)
                    elif not img_path.startswith('/'):
                        img_path = os.path.join(self.sensor_root, img_path)
                image_paths.append(img_path)
                
                # MapTR용 행렬 구성
                lidar2cam_rt = cam_info['lidar2cam_rt'] # 컨버터에서 계산한 값
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
            
            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
            ))
        
        # CAN Bus (VAD는 pkl에 저장된 can_bus를 씀)
        input_dict['can_bus'] = info.get('can_bus', np.zeros(18))
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        # 런타임 맵 생성 호출
        location = input_dict['map_location']
        e2g_t = input_dict['ego2global_translation']
        e2g_r = input_dict['ego2global_rotation']
        
        anns_results = self.vector_map.gen_vectorized_samples(location, e2g_t, e2g_r)
        
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        # LiDARInstanceLines 객체 전달
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc'] 
        
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        return example
    
    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        example = self.vectormap_pipeline(example, input_dict)
        return example