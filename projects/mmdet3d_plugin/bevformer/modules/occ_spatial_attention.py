
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16
import time
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from .spatial_cross_attention import MSDeformableAttention3D
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from .multi_scale_3ddeformable_attn_function import WeightedMultiScaleDeformableAttnFunction_fp32, \
    WeightedMultiScaleDeformableAttnFunction_fp16, MultiScaleDepthScoreSampleFunction_fp32, MultiScaleDepthScoreSampleFunction_fp16, \
    MultiScale3DDeformableAttnFunction_fp16, MultiScale3DDeformableAttnFunction_fp32
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class OccSpatialAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(OccSpatialAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img.reshape(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, 1, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                reference_points_per_img_new = reference_points_per_img.reshape(1,-1,1,2)
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img_new[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, 1, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.reshape(num_cams,bs,-1)
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

@ATTENTION.register_module()
class MSDeformableAttention3D_DFA3D(MSDeformableAttention3D):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8, im2col_step=64, dropout=0.1, batch_first=True, norm_cfg=None, init_cfg=None):
        super().__init__(embed_dims, num_heads, num_levels, num_points, im2col_step, dropout, batch_first, norm_cfg, init_cfg)
        self.sampling_offsets_depth = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 1)
        self.init_smpl_off_weights()
    def init_smpl_off_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets_depth, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([(thetas.cos() + thetas.sin()) / 2], -1)
        grid_init = grid_init.view(self.num_heads, 1, 1, 1).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets_depth.bias.data = grid_init.view(-1)
    def forward(self,
                query,
                key=None,
                value=None,
                value_dpt_dist=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        _, _, dim_depth = value_dpt_dist.shape
        value_dpt_dist = value_dpt_dist.view(bs, num_value, 1, dim_depth).repeat(1,1,self.num_heads, 1)
        sampling_offsets_uv = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        sampling_offsets_depth = self.sampling_offsets_depth(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 1)
        sampling_offsets = torch.cat([sampling_offsets_uv, sampling_offsets_depth], dim = -1)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        spatial_shapes_3D = self.get_spatial_shape_3D(spatial_shapes, dim_depth)
        if reference_points.shape[-1] == 3:

            offset_normalizer = torch.stack(
                [spatial_shapes_3D[..., 1], spatial_shapes_3D[..., 0], spatial_shapes_3D[..., 2]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            sampling_locations_ref = reference_points.repeat(1,1,num_heads,num_levels,num_points,1,1)
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            sampling_locations_ref = sampling_locations_ref.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp32
            output, depth_score = MultiScaleDeformableAttnFunction.apply(
                value, value_dpt_dist, spatial_shapes_3D, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        
        # weight_update is useful when self.use_empty == True.
        weight_update = (depth_score.mean(dim=-1) * attention_weights).flatten(-2).sum(dim=-1, keepdim=True)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output, weight_update
    def get_spatial_shape_3D(self, spatial_shape, depth_dim):
        spatial_shape_depth = spatial_shape.new_ones(*spatial_shape.shape[:-1], 1) * depth_dim
        spatial_shape_3D = torch.cat([spatial_shape, spatial_shape_depth], dim=-1)
        return spatial_shape_3D.contiguous()

@ATTENTION.register_module()
class OccSpatialAttention_DFA3D(OccSpatialAttention):
    def __init__(self, embed_dims=256, num_cams=6, bev_h=200, bev_w=200, bev_z=16, use_empty=False, num_head=8, pc_range=None, dropout=0.1, init_cfg=None, batch_first=False, deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_levels=4), **kwargs):
        super().__init__(embed_dims, num_cams, pc_range, dropout, init_cfg, batch_first, deformable_attention, **kwargs)
        self.bev_h = bev_h  # size of BEV. (height and width)
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_head = num_head  # num_head in deformable attention
        self.use_empty = use_empty  # if use empty tensor to fill the anchors that can not obtain any valid features.
        if use_empty:
            self.empty_query = nn.Embedding(self.bev_h*self.bev_w*self.bev_z*num_head, embed_dims)

    @force_fp32(apply_to=('query', 'key', 'value', 'value_dpt_dist', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                value_dpt_dist=None,
                flag='encoder',
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 3])
        empty_queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.num_head, self.embed_dims])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                if self.use_empty:
                    empty_queries_rebatch[j, i, :len(index_query_per_img)] = self.empty_query.weight.view(self.bev_h*self.bev_w*self.bev_z, self.num_head, self.embed_dims)[index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value_dpt_dist = value_dpt_dist.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, value_dpt_dist.shape[-1])

        queries, update_weight = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            value_dpt_dist=value_dpt_dist,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 3), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index, **kwargs)
        queries = queries.view(bs, self.num_cams, max_len, self.embed_dims)
        update_weight = update_weight.view(bs, self.num_cams, *update_weight.shape[1:])
        if self.use_empty:
            queries = queries + ((1-update_weight) * empty_queries_rebatch).mean(dim=-2)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.reshape(num_cams,bs,-1)
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

