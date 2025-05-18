from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_3D_custom_function import MultiScaleDeformableAttn3DCustomFunction_fp16, MultiScaleDeformableAttn3DCustomFunction_fp32
# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

@ATTENTION.register_module()
class OctreeSelfAttention3D(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas*0], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            3).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                octree_query,
                octree_geo_info=None,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points_list=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        assert octree_query.shape[0] == 1
        bev_h = [50,100,200]
        bev_w = [50,100,200]
        bev_z = [4,8,16]
        octree_geo_info_level = octree_geo_info[:, -1]

        output_list = []
        for i in range(3):
            reference_points = reference_points_list[i]
            level_mask = octree_geo_info_level == i
            query = octree_query[:,level_mask]
            spatial_shapes = torch.tensor([[bev_h[i], bev_w[i], bev_z[i]]], device=query.device)

            # if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

     
            identity = query
            
            if query_pos is not None:
                query = query + query_pos
            if not self.batch_first:
                query = query.permute(1, 0, 2)
                value = value.permute(1, 0, 2)
            bs,  num_query, embed_dims = query.shape

            _, num_value, _ = value.shape
            # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]* spatial_shapes[:, 2]).sum() == num_value
            assert self.num_bev_queue == 2

            query = torch.cat([value[:bs], query], -1)
            value = self.value_proj(value)

            assert key_padding_mask is None

            value = value.reshape(bs*self.num_bev_queue,
                                num_value, self.num_heads, -1)

            sampling_offsets = self.sampling_offsets(query)
            sampling_offsets = sampling_offsets.view(
                bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 3)
            attention_weights = self.attention_weights(query).view(
                bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.num_bev_queue,
                                                    self.num_levels,
                                                    self.num_points)

            attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
                .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
            sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
                .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 3)

            assert reference_points.shape[-1] == 3
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0],spatial_shapes[..., 2]], -1)#hwz
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

            if torch.cuda.is_available() and value.is_cuda:
                # using fp16 deformable attention is unstable because it performs many sum operations
                if value.dtype == torch.float16:
                    MultiScaleDeformableAttnFunction = MultiScaleDeformableAttn3DCustomFunction_fp16
                else:
                    MultiScaleDeformableAttnFunction = MultiScaleDeformableAttn3DCustomFunction_fp32

                output = MultiScaleDeformableAttnFunction.apply(
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                assert False
                # output = multi_scale_deformable_attn_pytorch(
                #     value, spatial_shapes, sampling_locations, attention_weights)

            output = output.permute(1, 2, 0)

            output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
            output = output.mean(-1)

            output = output.permute(2, 0, 1)

            output = self.output_proj(output)

            if not self.batch_first:
                output = output.permute(1, 0, 2)

            output_list.append(self.dropout(output) + identity)

        return torch.cat((output_list[0], output_list[1], output_list[2]), dim=1)



@ATTENTION.register_module()
class DeformSelfAttention3DCustom(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas*0], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            3).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):

        spatial_shapes=torch.tensor(
                        [[200,200, 16]], device=query.device)

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]* spatial_shapes[:, 2]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        assert key_padding_mask is None

        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 3)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 3)

        assert reference_points.shape[-1] == 3
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0],spatial_shapes[..., 2]], -1)#hwz
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]


        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttn3DCustomFunction_fp16
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttn3DCustomFunction_fp32


            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.permute(1, 2, 0)

        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity