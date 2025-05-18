import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg
from typing import Optional

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class OctreeOccTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 cam_encoder=None,
                 seg_decoder = None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 **kwargs):
        super(OctreeOccTransformer, self).__init__(**kwargs)
        self.cam_encoder = build_transformer_layer_sequence(cam_encoder)
        self.seg_decoder = build_transformer_layer_sequence(seg_decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'octree_queries', 'prev_bev', 'bev_pos'))
    def get_octree_features(
            self,
            mlvl_feats,
            octree_queries,
            octree_geo_info,
            bev_h,
            bev_w,
            bev_z,
            octree_mask=None,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            octree_pred_l1=None,
            octree_pred_l2=None,
            **kwargs):

        bs = mlvl_feats[0].size(0)
        octree_queries = octree_queries.unsqueeze(1).repeat(1, bs, 1)
        # bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # add can bus signals
        can_bus = octree_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        octree_queries = octree_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=octree_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        octree_embed, octree_pred_l1_block, octree_pred_l2_block, octree_structure_l1_block, octree_structure_l2_block, octree_mask_new, octree_geo_info_xyz, octree_geo_info_level = self.cam_encoder(
            octree_queries,
            octree_geo_info,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_z=bev_z,
            octree_mask=octree_mask,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            octree_pred_l1=octree_pred_l1,
            octree_pred_l2=octree_pred_l2,
            **kwargs
        )

        outputs = {}
        outputs['octree_feat'] = octree_embed
        outputs['octree_pred_l1_block'] = octree_pred_l1_block
        outputs['octree_pred_l2_block'] = octree_pred_l2_block
        outputs['octree_structure_l1_block'] = octree_structure_l1_block
        outputs['octree_structure_l2_block'] = octree_structure_l2_block
        outputs['octree_mask'] = octree_mask_new
        outputs['octree_geo_info_xyz'] = octree_geo_info_xyz
        outputs['octree_geo_info_level'] = octree_geo_info_level
  
        return outputs

    def align_prev_bev(self, prev_bev, bev_h, bev_w, bev_z, **kwargs):
        if prev_bev is not None:
            pc_range = self.cam_encoder.pc_range
            ref_y, ref_x, ref_z = torch.meshgrid(
                    torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_z - 0.5, bev_z, dtype=prev_bev.dtype, device=prev_bev.device),
                )
            ref_y = ref_y / bev_h
            ref_x = ref_x / bev_w
            ref_z = ref_z / bev_z

            GROUND_HEIGHT = -2
            grid = torch.stack(
                    (ref_x,
                    ref_y,
                    # ref_x.new_full(ref_x.shape, GROUND_HEIGHT),
                    ref_z,
                    ref_x.new_ones(ref_x.shape)), dim=-1)

            min_x, min_y, min_z, max_x, max_y, max_z = pc_range
            grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
            grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
            grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
            grid = grid.reshape(-1, 4)

            bs = prev_bev.shape[0]
            len_queue = prev_bev.shape[1]
            assert bs == 1
            for i in range(bs):
                lidar_to_ego = kwargs['img_metas'][i]['lidar2ego_transformation']
                curr_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][-1]

                curr_grid_in_prev_frame_lst = []
                for j in range(len_queue):
                    prev_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][j]
                    prev_lidar_to_curr_lidar =  np.linalg.inv(curr_ego_to_global) @ prev_ego_to_global 
                    curr_lidar_to_prev_lidar = np.linalg.inv(prev_lidar_to_curr_lidar)
                    curr_lidar_to_prev_lidar = grid.new_tensor(curr_lidar_to_prev_lidar)

                    curr_grid_in_prev_frame = torch.matmul(curr_lidar_to_prev_lidar, grid.T).T.reshape(bev_h, bev_w, bev_z, -1)[..., :3]
                    curr_grid_in_prev_frame[..., 0] = (curr_grid_in_prev_frame[..., 0] - min_x) / (max_x - min_x)
                    curr_grid_in_prev_frame[..., 1] = (curr_grid_in_prev_frame[..., 1] - min_y) / (max_y - min_y)
                    curr_grid_in_prev_frame[..., 2] = (curr_grid_in_prev_frame[..., 2] - min_z) / (max_z - min_z)
                    curr_grid_in_prev_frame = curr_grid_in_prev_frame * 2.0 - 1.0
                    curr_grid_in_prev_frame_lst.append(curr_grid_in_prev_frame)

                curr_grid_in_prev_frame = torch.stack(curr_grid_in_prev_frame_lst, dim=0)

                prev_bev_warp_to_curr_frame = torch.nn.functional.grid_sample(
                    prev_bev[i].permute(0, 1, 4, 2, 3),  # [bs, dim, z, h, w]
                    curr_grid_in_prev_frame.permute(0, 3, 1, 2, 4),  # [bs, z, h, w, 3]
                    align_corners=False)
                prev_bev = prev_bev_warp_to_curr_frame.permute(0, 1, 3, 4, 2).unsqueeze(0) # add bs dim, [bs, dim, h, w, z]
            return prev_bev

    def bev_temporal_fuse(
        self,
        bev_embeds: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        bev_h,
        bev_w,
        bev_z,
        **kwargs
    ) -> torch.Tensor:
        # [bs, num_queue, embed_dims, bev_h, bev_w]
        prev_bev = self.align_prev_bev(prev_bev, bev_h, bev_w, bev_z, **kwargs)

        bev_embeds = self.temporal_encoder(bev_embeds, prev_bev)

        return bev_embeds


    @auto_fp16(apply_to=('mlvl_feats', 'octree_queries', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                octree_queries,
                octree_geo_info,
                bev_h,
                bev_w,
                bev_z,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                prev_bev=None,
                octree_mask= None,
                octree_pred_l1=None,
                octree_pred_l2=None,
                **kwargs):
        outputs = {}

        out = self.get_octree_features(
            mlvl_feats,
            octree_queries,
            octree_geo_info,
            bev_h,
            bev_w,
            bev_z,
            octree_mask=octree_mask,
            grid_length=grid_length,
            bev_pos=bev_pos,
            octree_pred_l1=octree_pred_l1,
            octree_pred_l2=octree_pred_l2,
            **kwargs) 
        octree_feat = out.get('octree_feat', None)

        outputs['octree_feat'] = octree_feat
        outputs['octree_pred_l1_block'] = out.get('octree_pred_l1_block', None)
        outputs['octree_pred_l2_block'] = out.get('octree_pred_l2_block', None)
        outputs['octree_structure_l1_block'] = out.get('octree_structure_l1_block', None)
        outputs['octree_structure_l2_block'] = out.get('octree_structure_l2_block', None)

        octree_mask_new = out.get('octree_mask', None)
        octree_geo_info_xyz_new = out.get('octree_geo_info_xyz', None)
        octree_geo_info_level_new = out.get('octree_geo_info_level', None)
        octree_geo_info_new = torch.cat((octree_geo_info_xyz_new, octree_geo_info_level_new), dim=-1).to(octree_feat.device)

        occupancy, voxel_det = self.seg_decoder(octree_feat, octree_geo_info_new, octree_mask_new)
        occupancy = occupancy.permute(0,4,3,2,1)
            
        outputs['occ'] = occupancy
        outputs['voxel_det'] = voxel_det
        return outputs