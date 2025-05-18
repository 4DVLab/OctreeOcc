import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet3d.core import bbox3d2result, LiDARInstance3DBoxes
from projects.mmdet3d_plugin.models.losses import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.mmdet3d_plugin.models.losses import lovasz_softmax, CustomFocalLoss
from projects.mmdet3d_plugin.models.losses import nusc_class_frequencies, nusc_class_names

from ops import create_octree_mask_l1_to_l2 as cuda_create_mask_l1_to_l2
from ops import create_octree_mask_l2_to_l3 as cuda_create_mask_l2_to_l3

import math

@HEADS.register_module()
class OctreeOccHead(BaseModule):

    def __init__(self,
                 as_two_stage=False,
                 sync_cls_avg_factor=False,
                 transformer=None,
                 num_classes=18,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 bev_z=5,
                 loss_occ=None,
                 use_init_pred=False,
                 loss_octree_pred=None,
                 fixed_init_query=False,
                 use_mask=False,
                 use_octree_embed=False,
                 loss_occupancy_aux = None,
                 loss_det_occ = None,
                 use_self_attn=False,
                 octree_positional_encoding=None,
                 positional_encoding=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.num_classes=num_classes
        self.use_mask=use_mask
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_occupancy_aux = loss_occupancy_aux
        self.loss_det_occ = loss_det_occ
        self.loss_octree_pred = loss_octree_pred
        self.as_two_stage = as_two_stage
        self.use_init_pred = use_init_pred
        self.use_self_attn = use_self_attn
        self.use_octree_embed = use_octree_embed
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        super(OctreeOccHead, self).__init__()

        self.loss_occ = build_loss(loss_occ)
        if loss_occupancy_aux is not None:
            self.aux_loss = build_loss(loss_occupancy_aux)
        if loss_det_occ is not None:
            self.det_occ_loss = build_loss(loss_det_occ)
        if loss_octree_pred is not None:
            oct_prob = np.load('/public/home/luyh2/PanoOcc/projects/mmdet3d_plugin/models/utils/occupied_prob.npz')
            self.oct_prob_l1 = torch.from_numpy(oct_prob['octree_prob_l1']).unsqueeze(dim=0)
            self.oct_prob_l2 = torch.from_numpy(oct_prob['octree_prob_l2']).unsqueeze(dim=0)
            self.octree_pred_loss = VaniliaFocalLoss()
        # self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        
        self.fixed_init_query = fixed_init_query
        if self.fixed_init_query:
            self.octree_embedding = nn.Embedding(97500, self.embed_dims)
        else:
            self.octree_embedding = nn.Embedding(self.bev_h[0]*self.bev_w[0]*self.bev_z[0], self.embed_dims)
        if self.use_octree_embed:
            self.octree_positional_encoding = build_positional_encoding(octree_positional_encoding)
        if self.use_self_attn:
            self.positional_encoding = build_positional_encoding(positional_encoding)

        if self.use_init_pred:
            self.octree_predictor = nn.Sequential(
                nn.Conv3d(self.embed_dims, self.embed_dims // 8, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.embed_dims // 8),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.embed_dims // 8, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def get_real_xyz_per_query(self, mask_list):
        level = len(self.bev_h)
        xyz_list = []
        level_list = []
        
        device = mask_list[0].device
        dtype = torch.float32
        
        for i in range(level):
            h, w, z = self.bev_h[i], self.bev_w[i], self.bev_z[i]
            mask = mask_list[i][0]
            
            y_coords = torch.linspace(0.5/h, 1-0.5/h, h, device=device, dtype=dtype)
            x_coords = torch.linspace(0.5/w, 1-0.5/w, w, device=device, dtype=dtype)
            z_coords = torch.linspace(0.5/z, 1-0.5/z, z, device=device, dtype=dtype)
            
            ys, xs, zs = torch.meshgrid(y_coords, x_coords, z_coords)
            grid = torch.stack((xs.flatten(), ys.flatten(), zs.flatten()), dim=-1).view(h, w, z, 3)
            
            xyz_list.append(grid[mask])
            
            level_tensor = torch.ones_like(grid[..., 0:1], device=device) * i
            level_list.append(level_tensor[mask])
        
        return torch.cat(xyz_list, dim=0), torch.cat(level_list, dim=0)


    def create_octree_mask_l1_to_l2(self, octree_l1_to_l2):
        return cuda_create_mask_l1_to_l2(octree_l1_to_l2)

    def create_octree_mask_l2_to_l3(self, octree_l2_to_l3):
        return cuda_create_mask_l2_to_l3(octree_l2_to_l3)

    def octree_initialization(self, octree_structure_l1, octree_structure_l2):
        # octree_structure_l1: bs Y X Z

        octree_structure_l1 = octree_structure_l1.bool()
        octree_structure_l2 = octree_structure_l2.bool()

        mask = []
        mask_l1 = torch.logical_not(octree_structure_l1)
        num_feat_l1 = mask_l1.sum()
        mask.append(mask_l1)

        mask_l2_from_l1 = self.create_octree_mask_l1_to_l2(octree_structure_l1)
        mask_l2 = torch.where(mask_l2_from_l1, ~octree_structure_l2, torch.zeros_like(octree_structure_l2))
        num_feat_l2 = mask_l2.sum()
        mask.append(mask_l2)

        occ_ancestor_mask_l2 = torch.logical_and(mask_l2_from_l1, octree_structure_l2)
        assert torch.equal(occ_ancestor_mask_l2, octree_structure_l2)
        mask_l3_from_l2 = self.create_octree_mask_l2_to_l3(octree_structure_l2)
        num_feat_l3 = mask_l3_from_l2.sum()
        mask.append(mask_l3_from_l2)

        assert num_feat_l1 * 64 + num_feat_l2 * 8 + num_feat_l3 == 200*200*16

        return mask

    def create_irregular_quries(self, octree_query, mask):
        mask_l1, mask_l2, mask_l3 = mask[0], mask[1], mask[2]
    
        num_feat_l1 = mask_l1.sum()
        num_feat_l2 = mask_l2.sum()
        num_feat_l3 = mask_l3.sum()
        feat_dim = octree_query.shape[-1]
        
        result = torch.empty((num_feat_l1 + num_feat_l2 + num_feat_l3, feat_dim), 
                            dtype=octree_query.dtype, 
                            device=octree_query.device)
        
        octree_query = octree_query.view(1, self.bev_h[0], self.bev_w[0], self.bev_z[0], -1)
        
        result[:num_feat_l1] = octree_query[mask_l1]
        
        octree_query_perm = octree_query.permute(0, 4, 1, 2, 3)
        octree_query_l2_perm = F.interpolate(octree_query_perm, scale_factor=2, mode='trilinear')
        octree_query_l2 = octree_query_l2_perm.permute(0, 2, 3, 4, 1)
        result[num_feat_l1:num_feat_l1+num_feat_l2] = octree_query_l2[mask_l2]
        
        octree_query_l3_perm = F.interpolate(octree_query_l2_perm, scale_factor=2, mode='trilinear')
        octree_query_l3 = octree_query_l3_perm.permute(0, 2, 3, 4, 1)
        result[num_feat_l1+num_feat_l2:] = octree_query_l3[mask_l3]
        
        return result

    def octree_structure_init_pred(self, octree_query):
        octree_query = octree_query.view(1, self.bev_h[0], self.bev_w[0], self.bev_z[0], self.embed_dims).permute(0,4,1,2,3)
        octree_pred_l1 = self.octree_predictor(octree_query).permute(0,2,3,4,1).squeeze(dim=-1)
        octree_query_l2 = F.interpolate(octree_query, size=(100, 100, 8), mode='trilinear')
        octree_pred_l2 = self.octree_predictor(octree_query_l2).permute(0,2,3,4,1).squeeze(dim=-1)
        return octree_pred_l1 * self.oct_prob_l1.to(octree_pred_l1.device).permute(0,2,1,3), octree_pred_l2 * self.oct_prob_l2.to(octree_pred_l2.device).permute(0,2,1,3)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, test=False, subdividable_different_level_l1=None, subdividable_different_level_l2=None, seg_structure_l1=None, seg_structure_l2=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        octree_query = self.octree_embedding.weight.to(dtype)
        octree_pred_l1, octree_pred_l2 = self.octree_structure_init_pred(octree_query)
        octree_structure_l1 = seg_structure_l1[0]
        octree_structure_l2 = seg_structure_l2[0]
        assert octree_structure_l1.shape[0] == 1
        mask = self.octree_initialization(octree_structure_l1, octree_structure_l2)
        
        octree_geo_info_xyz, octree_geo_info_level = self.get_real_xyz_per_query(mask)
        octree_geo_info = torch.cat((octree_geo_info_xyz, octree_geo_info_level), dim=-1).to(mlvl_feats[0].device)
        
        octree_queries = self.create_irregular_quries(octree_query, mask)

        octree_pos = None

        if only_bev: 
            outputs = self.transformer.get_octree_features(
                mlvl_feats,
                octree_queries,
                octree_geo_info,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h[0],
                             self.real_w / self.bev_w[0]),
                octree_mask=mask,
                bev_pos=octree_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                octree_pred_l1=octree_pred_l1.clone(),
                octree_pred_l2=octree_pred_l2.clone(),
            )
            return outputs.get('octree_feat', None)
        else:
            outputs = self.transformer(
                mlvl_feats,
                octree_queries,
                octree_geo_info,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                output_queries=None,
                grid_length=(self.real_h / self.bev_h[0],
                             self.real_w / self.bev_w[0]),
                bev_pos=octree_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                octree_mask=mask,
                octree_pred_l1=octree_pred_l1.clone(),
                octree_pred_l2=octree_pred_l2.clone(),
            )

        outs = {
            'octree_embed': outputs.get("octree_feat", None),
            'occ':outputs.get("occ", None),
            'det_occ':outputs.get("voxel_det", None),
            'octree_pred_l1_block':outputs.get("octree_pred_l1_block", None),
            'octree_pred_l2_block':outputs.get("octree_pred_l2_block", None),
            'octree_pred_l1':octree_pred_l1,
            'octree_pred_l2':octree_pred_l2,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             voxel_semantics,
             subdividable_different_level_l1, 
             subdividable_different_level_l2,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()

        assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17

        occ = preds_dicts.get('occ', None)

        if occ is not None:
            losses = self.loss_single(voxel_semantics,mask_camera,occ)
            if self.loss_occupancy_aux is not None:
                occ_aux = occ.permute(0,4,1,2,3)
                losses_aux = self.aux_loss(occ_aux,voxel_semantics)
                loss_dict['loss_occ_aux']=losses_aux
            loss_dict['loss_occ']=losses
        else:
            assert False

        # add det occ
        det_occ = preds_dicts.get('det_occ', None)
        if det_occ is not None:
            voxel_det_mask = torch.ones(1,det_occ.shape[1],det_occ.shape[2],det_occ.shape[3]).to(occ.device).long()
            voxel_det = (voxel_semantics<17) & mask_camera.to(bool)
            Idx = torch.where(voxel_det)
            voxel_det_mask[Idx[0],Idx[1],Idx[2],Idx[3]]= 0
            if self.loss_det_occ is not None:
                num_total_pos = voxel_det.sum()
                avg_factor = num_total_pos*1.0
                if self.sync_cls_avg_factor:
                    avg_factor = reduce_mean(
                        det_occ.new_tensor([avg_factor]))
                avg_factor = max(avg_factor, 1)
                losses_det_occ = self.det_occ_loss(det_occ.reshape(-1, 1),voxel_det_mask.reshape(-1),avg_factor=avg_factor)
                loss_dict['loss_det_occ']=losses_det_occ

        octree_pred_l1 = preds_dicts.get('octree_pred_l1', None)
        if octree_pred_l1 is not None:
            subdividable_different_level_l1 = subdividable_different_level_l1.permute(0,2,1,3)
            losses_octree_pred_l1 = self.octree_pred_loss(octree_pred_l1, subdividable_different_level_l1.to(octree_pred_l1.dtype)) * 5.0
            loss_dict['loss_octree_pred_l1']=losses_octree_pred_l1

        octree_pred_l2 = preds_dicts.get('octree_pred_l2', None)
        if octree_pred_l2 is not None:
            subdividable_different_level_l2 = subdividable_different_level_l2.permute(0,2,1,3)
            losses_octree_pred_l2 = self.octree_pred_loss(octree_pred_l2, subdividable_different_level_l2.to(octree_pred_l2.dtype)) * 3.0
            loss_dict['loss_octree_pred_l2']=losses_octree_pred_l2 
        
        octree_pred_l1_block = preds_dicts.get('octree_pred_l1_block', None)
        if octree_pred_l1_block is not None:
            subdividable_different_level_l1 = subdividable_different_level_l1.permute(0,2,1,3)
            losses_octree_pred_l1_block = self.octree_pred_loss(octree_pred_l1_block, subdividable_different_level_l1.to(octree_pred_l1_block.dtype)) * 5.0
            loss_dict['loss_octree_pred_l1_block']=losses_octree_pred_l1_block

        octree_pred_l2_block = preds_dicts.get('octree_pred_l2_block', None)
        if octree_pred_l2_block is not None:
            subdividable_different_level_l2 = subdividable_different_level_l2.permute(0,2,1,3)
            losses_octree_pred_l2_block = self.octree_pred_loss(octree_pred_l2_block, subdividable_different_level_l2.to(octree_pred_l2_block.dtype)) * 3.0
            loss_dict['loss_octree_pred_l2_block']=losses_octree_pred_l2_block 

        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)

            loss_occ = self.loss_occ(preds, voxel_semantics)
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        occ_out=preds_dicts.get('occ', None)
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)

        # # # Post processing 1
        # det_occ = preds_dicts.get('det_occ', None)
        # if det_occ is not None:
        #     occupancy_mask = det_occ.sigmoid() < 0.05
        #     occupancy_mask = occupancy_mask.squeeze(-1)
        #     occ_score[occupancy_mask==1] = 17

        return occ_score       


class VaniliaFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(VaniliaFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')
        bce_loss = bce_loss(inputs, targets)

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()
