from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F
import torch
import time
import torch.distributed as dist
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MLP_Decoder(BaseModule):

    def __init__(self,
                 num_classes,
                 out_dim = 64,
                 inter_up_rate = [2,2,2],
                 upsampling_method='trilinear',
                 align_corners=False):
        super(MLP_Decoder, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.out_dim = out_dim
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
    
        self.mlp_decoder = MLP(dim_x=self.out_dim,act_fn='softplus',layer_size=2)
        self.classifier = nn.Linear(self.out_dim, self.num_classes)
                
    def forward(self, inputs):
        
        # z h w
        voxel_point = inputs.permute(0,2,3,4,1).view(1,-1,self.out_dim)
        voxel_point_feat = self.mlp_decoder(voxel_point)
        point_cls = self.classifier(voxel_point_feat)

        voxel_point_cls = point_cls.view(1,inputs.shape[2],inputs.shape[3],inputs.shape[4],-1).permute(0,4,1,2,3)

        voxel_logits = F.interpolate(voxel_point_cls,scale_factor=(self.inter_up_rate[0],self.inter_up_rate[1],self.inter_up_rate[2]),mode=self.upsampling_method,align_corners=self.align_corners)
        
        return voxel_logits

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OctreeDecoder(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_classes = 18,
                 out_dim=256,
                 bev_h = None,
                 bev_w = None,
                 bev_z = None,
                 output_size = None,
                 pc_range=None,):
        super(OctreeDecoder, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.out_dim = out_dim
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.output_size = output_size
        self.final_conv = nn.Sequential(
            nn.ConvTranspose3d(self.embed_dims, self.embed_dims, (3, 3, 3), stride=(1, 1, 1),padding=(1,1,1)),
            nn.BatchNorm3d(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.embed_dims, self.out_dim, (3, 3, 3), stride=(1, 1, 1),padding=(1,1,1)),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp_decoder = MLP(dim_x=self.out_dim,act_fn='softplus',layer_size=2)
        self.classifier = nn.Linear(self.out_dim, self.num_classes)
        self.semantic_det = nn.Sequential(nn.Conv3d(self.out_dim, 1, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def assign_feat_to_final_size(self, octree_feat, octree_geo_info_xyz, octree_geo_info_level, mask):
        octree_geo_info_level = octree_geo_info_level.squeeze()

        output_feat = torch.zeros(1, 50, 50, 4, self.embed_dims).to(device=octree_feat.device)

        mask_l1 = mask[0]
        mask_l2 = mask[1]
        mask_l3 = mask[2]

        index_l1 = torch.nonzero(mask_l1)
        index_l2 = torch.nonzero(mask_l2)
        index_l3 = torch.nonzero(mask_l3)

        level_mask_l1 = octree_geo_info_level == 0
        level_mask_l2 = octree_geo_info_level == 1
        level_mask_l3 = octree_geo_info_level == 2
        
        octree_feat_l1 = octree_feat[level_mask_l1]
        octree_feat_l2 = octree_feat[level_mask_l2]
        octree_feat_l3 = octree_feat[level_mask_l3]

        #level 1
        # leaf_node_l1 = torch.where(mask_l1 == True)
        output_feat[index_l1[:, 0], index_l1[:, 1], index_l1[:, 2], index_l1[:, 3]] = octree_feat_l1

        # level 2
        output_feat = F.interpolate(output_feat.permute(0,4,1,2,3), scale_factor=2, mode='trilinear').permute(0,2,3,4,1)
        # leaf_node_l2 = torch.where(mask_l2 == True)
        output_feat[index_l2[:, 0], index_l2[:, 1], index_l2[:, 2], index_l2[:, 3]] = octree_feat_l2

        # level 3
        output_feat = F.interpolate(output_feat.permute(0,4,1,2,3), scale_factor=2, mode='trilinear').permute(0,2,3,4,1)
        # leaf_node_l3 = torch.where(mask_l3 == True)
        output_feat[index_l3[:, 0], index_l3[:, 1], index_l3[:, 2], index_l3[:, 3]] = octree_feat_l3

        return output_feat

    def forward(self, octree_feat, octree_geo_info, mask):
        octree_feat = octree_feat.squeeze()
        
        octree_geo_info_xyz = octree_geo_info[:, 0:-1]
        octree_geo_info_level = octree_geo_info[:, -1]

        voxel_point = self.assign_feat_to_final_size(octree_feat, octree_geo_info_xyz, octree_geo_info_level, mask)

        voxel_point = voxel_point.permute(0,4,3,1,2)

        voxel_point = self.final_conv(voxel_point)

        voxel_det = self.semantic_det(voxel_point)

        mlp_input = voxel_point.permute(0,2,3,4,1).view(1,-1,self.out_dim)

        voxel_point_feat = self.mlp_decoder(mlp_input)
        point_cls = self.classifier(voxel_point_feat)

        voxel_logits = point_cls.view(1,voxel_point.shape[2],voxel_point.shape[3],voxel_point.shape[4],-1).permute(0,4,1,2,3)
        
        return voxel_logits, voxel_det.permute(0,4,3,2,1)
        # return voxel_logits



class MLP(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == 'softplus':
                self.nn_layers.append(torch.nn.Softplus())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == 'softplus':
                    self.nn_layers.append(torch.nn.Softplus())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x
