
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.utils import build_from_cfg
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import ext_loader
import torch.distributed as dist
from ops import create_octree_mask_l1_to_l2 as cuda_create_mask_l1_to_l2
from ops import create_octree_mask_l2_to_l3 as cuda_create_mask_l2_to_l3

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OctreeOccupancyEncoder(TransformerLayerSequence):
    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes', ego=False,
                 **kwargs):
        super(OctreeOccupancyEncoder_new_self_attn, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.ego = ego 

        self.bev_h = [50,100,200]
        self.bev_w = [50,100,200]
        self.bev_z = [4,8,16]

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

    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            if self.ego:
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                    device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                    device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                    device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            else:
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                    device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                    device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                    device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x, ref_z = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                torch.linspace(
                    0.5, Z - 0.5, Z, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_z = ref_z.reshape(-1)[None] / Z
            ref_2d = torch.stack((ref_x, ref_y, ref_z), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    
        if self.ego:
            ego2lidar=img_metas[0]['ego2lidar']
            ego2lidar = reference_points.new_tensor(ego2lidar)
    
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # print(reference_points.size()[:3])
        # print(D, B)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        assert B == 1

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        
        if self.ego:
            ego2lidar=ego2lidar.view(1,1,1,1,4,4).repeat(D,1,num_cam,num_query,1,1)
            reference_points_cam = torch.matmul(torch.matmul(lidar2img.to(torch.float32),ego2lidar.to(torch.float32)),reference_points.to(torch.float32)).squeeze(-1)
        else:
            reference_points_cam = torch.matmul(lidar2img.to(torch.float32),reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask


    def create_octree_mask_l1_to_l2(self, octree_l1_to_l2):
        return cuda_create_mask_l1_to_l2(octree_l1_to_l2)

    def create_octree_mask_l2_to_l3(self, octree_l2_to_l3):
        return cuda_create_mask_l2_to_l3(octree_l2_to_l3)

    def get_ref_and_mask(self, reference_points_cam_list, bev_mask_list, octree_mask):
        reference_points_cam_list_after = []
        occ_mask_list_after = []
        for i in range(3):
            mask_level = octree_mask[i].squeeze()
            reference_points_cam = reference_points_cam_list[i][mask_level]
            reference_points_cam_list_after.append(reference_points_cam)
            bev_mask = bev_mask_list[i][mask_level]
            occ_mask_list_after.append(bev_mask)
        reference_points_cam = torch.cat((reference_points_cam_list_after[0], reference_points_cam_list_after[1], reference_points_cam_list_after[2]),dim=0).permute(1, 2, 0, 3)
        occ_mask = torch.cat((occ_mask_list_after[0], occ_mask_list_after[1], occ_mask_list_after[2]),dim=0).permute(1, 2, 0).contiguous()

        return reference_points_cam, occ_mask
    
    def convert_octree_prob_to_bool(self, octree_pred_l1, octree_pred_l2, ratio_l1=0.2, ratio_l2=0.6):
        total_elements = octree_pred_l1.numel()
        num_true_elements_l1 = int(ratio_l1 * total_elements)
        
        values, indices = torch.topk(octree_pred_l1.flatten(), num_true_elements_l1, largest=True)
        
        octree_structure_l1 = torch.zeros_like(octree_pred_l1, dtype=torch.bool)
        octree_structure_l1.view(-1)[indices] = True
        
        octree_structure_l2_from_l1 = self.create_octree_mask_l1_to_l2(octree_structure_l1)
        
        octree_pred_l2_masked = octree_pred_l2.clone()
        octree_pred_l2_masked[~octree_structure_l2_from_l1] = -1
        
        num_true_elements_l2 = int(int(8 * ratio_l1 * total_elements) * ratio_l2)
        
        values, indices = torch.topk(octree_pred_l2_masked.flatten(), num_true_elements_l2, largest=True)
        
        octree_structure_l2 = torch.zeros_like(octree_pred_l2, dtype=torch.bool)
        octree_structure_l2.view(-1)[indices] = True
        
        return octree_structure_l1, octree_structure_l2
    
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

    def octree_structure_pred(self, octree_query, octree_pred_l1_ori, octree_pred_l2_ori, from_init=False):
        octree_query_perm = octree_query.view(1, self.bev_h[0], self.bev_w[0], self.bev_z[0], self.embed_dims).permute(0,4,1,2,3)
        
        octree_pred_l1 = self.octree_predictor(octree_query_perm)
        octree_query_l2 = F.interpolate(octree_query_perm, size=(100, 100, 8), mode='trilinear')
        octree_pred_l2 = self.octree_predictor(octree_query_l2)
        
        octree_pred_l1 = octree_pred_l1.permute(0,2,3,4,1).squeeze(dim=-1)
        octree_pred_l2 = octree_pred_l2.permute(0,2,3,4,1).squeeze(dim=-1)
        
        ratio_keep = 0.5 if from_init else 0.3
        
        octree_pred_l1.mul_(ratio_keep).add_(octree_pred_l1_ori * ratio_keep)
        octree_pred_l2.mul_(ratio_keep).add_(octree_pred_l2_ori * ratio_keep)
        
        return octree_pred_l1, octree_pred_l2
    
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

    def assign_feat_to_top(self, octree_feat, octree_geo_info_level, mask):
        octree_feat = octree_feat.squeeze()
        octree_geo_info_level = octree_geo_info_level.squeeze()
        
        output_feat = torch.zeros(1, 200, 200, 16, self.embed_dims, device=octree_feat.device)
        
        level_mask_l1 = (octree_geo_info_level == 0)
        level_mask_l2 = (octree_geo_info_level == 1)
        level_mask_l3 = (octree_geo_info_level == 2)
        
        octree_feat_l1 = octree_feat[level_mask_l1]
        octree_feat_l2 = octree_feat[level_mask_l2]
        octree_feat_l3 = octree_feat[level_mask_l3]
        
        index_l1 = torch.nonzero(mask[0], as_tuple=True)
        index_l2 = torch.nonzero(mask[1], as_tuple=True)
        index_l3 = torch.nonzero(mask[2], as_tuple=True)
        
        for i, idx in enumerate(zip(*index_l1)):
            y, x, z = idx[1], idx[2], idx[3]
            output_feat[0, y*4:(y+1)*4, x*4:(x+1)*4, z*4:(z+1)*4] = octree_feat_l1[i]

        for i, idx in enumerate(zip(*index_l2)):
            y, x, z = idx[1], idx[2], idx[3]
            output_feat[0, y*2:(y+1)*2, x*2:(x+1)*2, z*2:(z+1)*2] = octree_feat_l2[i]

        output_feat[0, index_l3[1], index_l3[2], index_l3[3]] = octree_feat_l3
        
        return F.interpolate(output_feat.permute(0,4,1,2,3), size=(50,50,4), mode='trilinear').permute(0,2,3,4,1)


    def update_octree_structure(self, output, octree_geo_info_level, octree_mask, octree_pred_l1_ori, octree_pred_l2_ori, lid, reference_points_cam_list, bev_mask_list):
        feat_top_level = self.assign_feat_to_top(output, octree_geo_info_level, octree_mask)
        octree_pred_l1, octree_pred_l2 = self.octree_structure_pred(feat_top_level, octree_pred_l1_ori, octree_pred_l2_ori, from_init=lid==0)
        
        with torch.no_grad():
            octree_structure_l1, octree_structure_l2 = self.convert_octree_prob_to_bool(
                octree_pred_l1.detach(), octree_pred_l2.detach(), ratio_l1=0.2, ratio_l2=0.6)
            octree_mask_new = self.octree_initialization(octree_structure_l1, octree_structure_l2)
        
        octree_geo_info_xyz_new, octree_geo_info_level_new = self.get_real_xyz_per_query(octree_mask_new)
        
        outputs = {
            'octree_query': self.create_irregular_quries_top_down_mode(feat_top_level, octree_mask_new).unsqueeze(0),
            'octree_pred_l1': octree_pred_l1,
            'octree_pred_l2': octree_pred_l2,
            'octree_structure_l1': octree_structure_l1,
            'octree_structure_l2': octree_structure_l2,
            'octree_mask': octree_mask_new,
            'octree_geo_info_level': octree_geo_info_level_new,
            'octree_geo_info_xyz': octree_geo_info_xyz_new
        }
        
        reference_points_cam, occ_mask = self.get_ref_and_mask(reference_points_cam_list, bev_mask_list, octree_mask_new)
        outputs['reference_points_cam'] = reference_points_cam
        outputs['occ_mask'] = occ_mask
        
        return outputs
    
    @auto_fp16()
    def forward(self,
                octree_query,
                octree_geo_info,
                key,
                value,
                *args,
                octree_mask=None,
                bev_h=None,
                bev_w=None,
                bev_z=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                octree_pred_l1=None,
                octree_pred_l2=None,
                **kwargs):
        output = octree_query
        intermediate = []
        octree_geo_info_xyz = octree_geo_info[:, 0:-1]
        octree_geo_info_level = octree_geo_info[:, -1]
        reference_points_cam_list = []
        occ_mask_list = []
        ref_2d_list = []

        for i in range(len(bev_h)):
            mask_level = octree_mask[i].squeeze()
            ref_3d = self.get_reference_points(bev_h[i], bev_w[i], bev_z[i], bev_z[i], dim='3d', bs=octree_query.size(1),  device=octree_query.device, dtype=octree_query.dtype)
            reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['img_metas'])
            reference_points_cam = reference_points_cam.view(6, 1, bev_h[i], bev_w[i], bev_z[i], 2).permute(2, 3, 4, 0, 1, 5)
            reference_points_cam = reference_points_cam[mask_level]
            reference_points_cam_list.append(reference_points_cam)
            bev_mask = bev_mask.view(6, 1, bev_h[i], bev_w[i], bev_z[i]).permute(2, 3, 4, 0, 1)
            bev_mask = bev_mask[mask_level]
            occ_mask_list.append(bev_mask)
            
            ref_2d = self.get_reference_points(bev_h[i], bev_w[i], bev_z[i], bev_z[i], dim='2d', bs=octree_query.size(1), device=octree_query.device, dtype=octree_query.dtype)
            ref_2d = ref_2d.reshape(1, bev_h[i], bev_w[i], bev_z[i], 1, 3)
            ref_2d = ref_2d[:, mask_level]  # bs, num_query, 1, 3
            bs, len_bev, num_bev_level, _ = ref_2d.shape
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 3)
            ref_2d_list.append(hybird_ref_2d)

        reference_points_cam = torch.cat((reference_points_cam_list[0], reference_points_cam_list[1], reference_points_cam_list[2]),dim=0).permute(1, 2, 0, 3)
        occ_mask = torch.cat((occ_mask_list[0], occ_mask_list[1], occ_mask_list[2]),dim=0).permute(1, 2, 0).contiguous()
        
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        octree_query = octree_query.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            if lid == 0:
                output = layer(
                    octree_query,
                    key,
                    value,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=ref_2d_list,
                    ref_3d=None,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_z=bev_z,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam,
                    bev_mask=occ_mask,
                    octree_mask=octree_mask,
                    prev_bev=prev_bev,
                    octree_geo_info=octree_geo_info,
                    **kwargs)

                outputs = self.update_octree_structure(output, octree_geo_info_level, octree_mask, octree_pred_l1, octree_pred_l2, lid, reference_points_cam_list, bev_mask_list)

                octree_query = outputs['octree_query']
                octree_pred_l1_lid_0 = outputs['octree_pred_l1']
                octree_pred_l2_lid_0 = outputs['octree_pred_l2']
                octree_structure_l1_lid_0 = outputs['octree_structure_l1']
                octree_structure_l2_lid_0 = outputs['octree_structure_l2']
                octree_geo_info_level_lid_0 = outputs['octree_geo_info_level']
                octree_geo_info_xyz_lid_0 = outputs['octree_geo_info_xyz']
                reference_points_cam_lid_0 = outputs['reference_points_cam']
                octree_mask_lid_0 = outputs['octree_mask']
                occ_mask_lid_0 = outputs['occ_mask']

            elif lid == 1:
                output = layer(
                    octree_query,
                    key,
                    value,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=ref_2d_list,
                    ref_3d=None,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_z=bev_z,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam_lid_0,
                    bev_mask=occ_mask_lid_0,
                    octree_mask=octree_mask,
                    prev_bev=prev_bev,
                    octree_geo_info=octree_geo_info,
                    **kwargs)

                outputs = self.update_octree_structure(output, octree_geo_info_level_lid_0, octree_mask_lid_0, octree_pred_l1_lid_0, octree_pred_l2_lid_0, lid, reference_points_cam_list, bev_mask_list)

                octree_query = outputs['octree_query']
                octree_pred_l1_lid_1 = outputs['octree_pred_l1']
                octree_pred_l2_lid_1 = outputs['octree_pred_l2']
                octree_structure_l1_lid_1 = outputs['octree_structure_l1']
                octree_structure_l2_lid_1 = outputs['octree_structure_l2']
                octree_geo_info_level_lid_1 = outputs['octree_geo_info_level']
                octree_geo_info_xyz_lid_1 = outputs['octree_geo_info_xyz']
                reference_points_cam_lid_1 = outputs['reference_points_cam']
                octree_mask_lid_1 = outputs['octree_mask']
                occ_mask_lid_1 = outputs['occ_mask']
            
            elif lid == 2 or lid == 3:
                output = layer(
                    octree_query,
                    key,
                    value,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=ref_2d_list,
                    ref_3d=None,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    bev_z=bev_z,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam_lid_1,
                    bev_mask=occ_mask_lid_1,
                    octree_mask=octree_mask,
                    prev_bev=prev_bev,
                    octree_geo_info=octree_geo_info,
                    **kwargs)

                octree_query = output
            else:
                assert False
        
        return output, octree_pred_l1_lid_1, octree_pred_l2_lid_1, octree_structure_l1_lid_1, octree_structure_l2_lid_1, octree_mask_lid_1, octree_geo_info_xyz_lid_1, octree_geo_info_level_lid_1


@TRANSFORMER_LAYER.register_module()
class OctreeOccupancyLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OctreeOccupancyLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) <= 6
        assert set(operation_order) <= set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                bev_z=None,
                reference_points_cam=None,
                mask=None,
                octree_mask=None,
                octree_geo_info=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    key=prev_bev,
                    value=prev_bev,
                    identity=identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points_list=ref_2d,
                    octree_geo_info=octree_geo_info,
                    spatial_shapes=torch.tensor([
                              [bev_z[0], bev_h[0], bev_w[0]],
                              [bev_z[1], bev_h[1], bev_w[1]],
                              [bev_z[2], bev_h[2], bev_w[2]]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

