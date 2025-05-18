from mmdet.datasets.builder import PIPELINES
import numpy as np
import os
from numpy import random
import mmcv
from mmcv.parallel import DataContainer as DC
import torch

@PIPELINES.register_module()
class LoadDenseLabel(object):
    def __init__(self,grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],occupancy_root=None):
        self.grid_size = np.array(grid_size)
        self.pc_range = pc_range
        self.occupancy_root = occupancy_root

    
    def __call__(self, results):

        scene_token = results['scene_token']
        lidar_token = results['lidar_token']

        occupancy_path = os.path.join(self.occupancy_root,'scene_'+scene_token,'occupancy',lidar_token+'.npy')
        
        # [z,y,x,label]
        occupancy_data = np.load(occupancy_path)

        results['dense_occupancy'] = occupancy_data

        return results



@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        if 'occ_gt_path' in results:
             occ_gt_path = results['occ_gt_path']
             occ_gt_path = os.path.join(self.data_root,occ_gt_path)

             occ_labels = np.load(occ_gt_path)
             semantics = occ_labels['semantics']
             mask_lidar = occ_labels['mask_lidar']
             mask_camera = occ_labels['mask_camera']
        else:
             semantics = np.zeros((200,200,16),dtype=np.uint8)
             mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
             mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

@PIPELINES.register_module()
class LoadMultiViewSegLabelFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size     = map_size

    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2]

        img_paths = results['img_filename']
        img_gt_list = []
        for img_path in img_paths:
            img_gt = np.load(img_path)['img_sem_gt']
            img_gt = np.expand_dims(img_gt, axis=-1)
            img_gt_list.append(img_gt)
        img_seg_gt = np.stack(img_gt_list, axis=-1)
        # seg_gt shape: (num_view, h, w)

        results['seg_gt'] = [img_seg_gt[..., i] for i in range(img_seg_gt.shape[-1])]
        return results

@PIPELINES.register_module()
class LoadSegPriorFromFile(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            seg_struct_path = os.path.join(self.data_root,occ_gt_path).replace("labels.npz", "oct_struct_from_seg_prior.npz")
            seg_struct = np.load(seg_struct_path)
            octree_structure_l1_from_seg = seg_struct['octree_structure_l1']
            octree_structure_l2_from_seg = seg_struct['octree_structure_l2']
        else:
            assert False
        results['seg_structure_l1'] = octree_structure_l1_from_seg
        results['seg_structure_l2'] = octree_structure_l2_from_seg
        return results

@PIPELINES.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size     = map_size
    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2]
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth)
                map_depths.append(map_depth)
        # img is of shape (h, w, c, num_views)
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """


        depth_coords = point_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(self.map_size)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
        
@PIPELINES.register_module()
class LoadOccGTFromFile_octree(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        if 'occ_gt_path' in results:
             occ_gt_path = results['occ_gt_path']
             occ_gt_path = os.path.join(self.data_root,occ_gt_path)

             occ_labels = np.load(occ_gt_path)
             semantics = occ_labels['semantics']
             mask_lidar = occ_labels['mask_lidar']
             mask_camera = occ_labels['mask_camera']
        else:
             semantics = np.zeros((200,200,16),dtype=np.uint8)
             mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
             mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        if 'occ_octree_gt' in results:
            occ_octree_gt = results['occ_octree_gt']
            occ_octree_path = os.path.join(self.data_root,occ_octree_gt)
            if os.path.exists(occ_octree_path):
                occ_octree_gt = np.load(occ_octree_path,allow_pickle=True)
            else:
                print(occ_octree_path)
                assert False
            subdividable_different_level_l1 = occ_octree_gt['occupied_gt_l1']
            subdividable_different_level_l2 = occ_octree_gt['occupied_gt_l2']
        else:
            subdividable_different_level_l1 = np.zeros((50, 50, 4), dtype=np.uint8)
            subdividable_different_level_l2 = np.zeros((100, 100, 8), dtype=np.uint8)
            
        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        results['subdividable_different_level_l1'] = subdividable_different_level_l1
        results['subdividable_different_level_l2'] = subdividable_different_level_l2

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            ignore_nonvisible=False,
            mask='mask_camera',
            ignore_classes=[],
            fix_void=True
        ):
        self.data_root = data_root
        self.ignore_nonvisible = ignore_nonvisible
        self.mask = mask
        self.ignore_classes=ignore_classes
        self.fix_void = fix_void

    def __call__(self, results):
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root,occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']

            occupancy = torch.tensor(semantics)
            visible_mask = torch.tensor(mask_camera)
            if self.ignore_nonvisible:
                occupancy[~visible_mask.to(torch.bool)] = 255
            
            if self.fix_void:
                occupancy[occupancy<255] = occupancy[occupancy<255] + 1

            for class_ in self.ignore_classes:
                occupancy[occupancy==class_] = 255
        else:
             semantics = np.zeros((200,200,16),dtype=np.uint8)
             mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
             mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['voxel_semantics'] = occupancy
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)