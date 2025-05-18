import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask, GridMaskHybrid
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmdet.models import builder
import torch.distributed as dist
import os
      
@DETECTORS.register_module()
class OctreeOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 occupancy_save_path=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 time_interval=1,
                 ):

        super(OctreeOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        for param in self.pts_bbox_head.parameters():
            param.requires_grad = True
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.time_interval = time_interval
        self.occupancy_save_path = occupancy_save_path
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': [],
            "ego2global_transform_lst": [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())

        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'),out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics,
                          mask_camera,
                          img_metas,
                          subdividable_different_level_l1=None, 
                          subdividable_different_level_l2=None, 
                          seg_structure_l1=None,
                          seg_structure_l2=None,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        # not suitable for online evaluation because the segmentation results generated and stored offline are used.
        # subdividable_different_level_l1 & l2: octree gt
        # seg_structure_l1 & l2: initial octree structure from segmentation model, details about how to obtainsee utils
        assert subdividable_different_level_l1 is not None
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, subdividable_different_level_l1=subdividable_different_level_l1, subdividable_different_level_l2=subdividable_different_level_l2, seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, voxel_semantics,
                       subdividable_different_level_l1, subdividable_different_level_l2,
                       mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list, subdividable_different_level_l1=None, subdividable_different_level_l2=None, seg_structure_l1=None, seg_structure_l2=None,):
        assert subdividable_different_level_l1 is not None
        is_training = self.training
        self.eval()

        prev_bev_lst = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, only_bev=True, 
                    subdividable_different_level_l1=subdividable_different_level_l1, subdividable_different_level_l2=subdividable_different_level_l2, seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2)
                prev_bev = prev_bev.permute(0, 2, 1)
                if prev_bev.shape[-1] == 10000:
                    prev_bev = prev_bev.reshape(prev_bev.shape[0], -1, self.pts_bbox_head.bev_h[0], self.pts_bbox_head.bev_w[0], self.pts_bbox_head.bev_z[0])
                prev_bev_lst.append(prev_bev)
        if is_training:
            self.train()

        return torch.stack(prev_bev_lst, dim=1)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      subdividable_different_level_l1=None, 
                      subdividable_different_level_l2=None, 
                      seg_structure_l1=None,
                      seg_structure_l2=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if prev_img.size(1)==0:
            prev_bev = None
        else:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, subdividable_different_level_l1=subdividable_different_level_l1, 
                                               subdividable_different_level_l2=subdividable_different_level_l2,
                                               seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, voxel_semantics, mask_camera, img_metas,
                                            subdividable_different_level_l1=subdividable_different_level_l1, subdividable_different_level_l2=subdividable_different_level_l2, 
                                            seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2,
                                            gt_bboxes_ignore=gt_bboxes_ignore, prev_bev=prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,
                     subdividable_different_level_l1=None, 
                     subdividable_different_level_l2=None, 
                     seg_structure_l1=None,
                     seg_structure_l2=None,
                     **kwargs):
        if isinstance(subdividable_different_level_l1, list):
            subdividable_different_level_l1 = subdividable_different_level_l1[0]
            subdividable_different_level_l2 = subdividable_different_level_l2[0]

        if isinstance(seg_structure_l1, list):
            seg_structure_l1 = seg_structure_l1[0]
            seg_structure_l2 = seg_structure_l2[0]

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        self.prev_frame_info["ego2global_transformation_lst"].append(img_metas[0][0]["ego2global_transformation"])

        img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"][-1::-self.time_interval][::-1]
        prev_bev = self.prev_frame_info['prev_bev'][-self.time_interval:: -self.time_interval][:: -1]

        # prev_bev = torch.stack(prev_bev, dim=1) if len(prev_bev) > 0 else None
        prev_bev = prev_bev[0] if len(prev_bev) > 0 else None

        new_prev_bev, occ_results = self.simple_test(
            img_metas[0], img[0], prev_bev=None, subdividable_different_level_l1=subdividable_different_level_l1, 
            seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2,
            subdividable_different_level_l2=subdividable_different_level_l2, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        new_prev_bev = new_prev_bev.permute(0, 2, 1)
        self.prev_frame_info['prev_bev'].append(new_prev_bev)

        while len(self.prev_frame_info["prev_bev"]) >= 4 * self.time_interval:
            self.prev_frame_info["prev_bev"].pop(0)
            self.prev_frame_info["ego2global_transformation_lst"].pop(0)

        return occ_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, subdividable_different_level_l1=None, subdividable_different_level_l2=None, seg_structure_l1=None, seg_structure_l2=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, test=True, subdividable_different_level_l1=subdividable_different_level_l1, 
               subdividable_different_level_l2=subdividable_different_level_l2, seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return outs.get('octree_embed', None), occ

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, subdividable_different_level_l1=None, subdividable_different_level_l2=None, seg_structure_l1=None, seg_structure_l2=None):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, subdividable_different_level_l1=subdividable_different_level_l1, 
            subdividable_different_level_l2=subdividable_different_level_l2, seg_structure_l1=seg_structure_l1, seg_structure_l2=seg_structure_l2, rescale=rescale)

        return new_prev_bev, occ
