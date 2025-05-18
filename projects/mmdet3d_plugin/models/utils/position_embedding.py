import torch
import torch.nn as nn
import math
from mmcv.cnn import uniform_init
from mmdet.models.utils.positional_encoding import POSITIONAL_ENCODING

class RelPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, pos_norm=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.fc = nn.Linear(4, self.num_pos_feats,bias=False)
        #nn.init.orthogonal_(self.fc.weight)
        #self.fc.weight.requires_grad = False
        self.pos_norm = pos_norm
        if self.pos_norm:
            self.norm = nn.LayerNorm(self.num_pos_feats)
    def forward(self, tensor):
        #mask = nesttensor.mask
        B,C,H,W = tensor.shape
        #print('tensor.shape',  tensor.shape)
        y_range = (torch.arange(H) / float(H - 1)).to(tensor.device)
        #y_axis = torch.stack((y_range, 1-y_range),dim=1)
        y_axis = torch.stack((torch.cos(y_range * math.pi), torch.sin(y_range * math.pi)), dim=1)
        y_axis = y_axis.reshape(H, 1, 2).repeat(1, W, 1).reshape(H * W, 2)

        x_range = (torch.arange(W) / float(W - 1)).to(tensor.device)
        #x_axis =torch.stack((x_range,1-x_range),dim=1)
        x_axis = torch.stack((torch.cos(x_range * math.pi), torch.sin(x_range * math.pi)), dim=1)
        x_axis = x_axis.reshape(1, W, 2).repeat(H, 1, 1).reshape(H * W, 2)
        x_pos = torch.cat((y_axis, x_axis), dim=1)
        x_pos = self.fc(x_pos)

        if self.pos_norm:
            x_pos = self.norm(x_pos)
        #print('xpos,', x_pos.max(),x_pos.min())
        return x_pos



@POSITIONAL_ENCODING.register_module()
class Learned3DPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, z_num_embed=8):
        super(Learned3DPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, 2*num_feats)
        self.col_embed = nn.Embedding(col_num_embed, 2*num_feats)
        self.z_embed = nn.Embedding(z_num_embed, 2*num_feats)
        self.num_feats = 2*num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.z_num_embed = z_num_embed
        self.init_weights()

    def init_weights(self):
        """Initialize the learnable weights."""
        uniform_init(self.row_embed)
        uniform_init(self.col_embed)
        uniform_init(self.z_embed)

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w, z].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """

        h, w, z = mask.shape[-3:]
        x = torch.arange(h, device=mask.device)
        y = torch.arange(w, device=mask.device)
        zz = torch.arange(z, device=mask.device)

        x_embed = self.row_embed(x)
        y_embed = self.col_embed(y)
        zz_embed = self.z_embed(zz)

        pos3d = x_embed.view(h,1,1,self.num_feats).expand(h,w,z,self.num_feats) + y_embed.view(1,w,1,self.num_feats).expand(h,w,z,self.num_feats) + zz_embed.view(1,1,z,self.num_feats).expand(h,w,z,self.num_feats)
        
        pos3d = pos3d.permute(3, 0, 1, 2).unsqueeze(0)
        
        return pos3d

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        repr_str += f'z_num_embed={self.z_num_embed})'
        return repr_str

@POSITIONAL_ENCODING.register_module()
class OctreePositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=256, pos_norm=True):
        super().__init__()
        self.occ_h = [50, 100, 200]
        self.occ_w = [50, 100, 200]
        self.occ_z = [4,8,16]
        self.row_embed_l1 = nn.Embedding(self.occ_h[0], num_pos_feats)
        self.col_embed_l1 = nn.Embedding(self.occ_w[0], num_pos_feats)
        self.z_embed_l1 = nn.Embedding(self.occ_z[0], num_pos_feats)
        self.row_embed_l2 = nn.Embedding(self.occ_h[1], num_pos_feats)
        self.col_embed_l2 = nn.Embedding(self.occ_w[1], num_pos_feats)
        self.z_embed_l2 = nn.Embedding(self.occ_z[1], num_pos_feats)
        self.row_embed_l3 = nn.Embedding(self.occ_h[2], num_pos_feats)
        self.col_embed_l3 = nn.Embedding(self.occ_w[2], num_pos_feats)
        self.z_embed_l3 = nn.Embedding(self.occ_z[2], num_pos_feats)
        self.num_feats = num_pos_feats

        self.pos_norm = pos_norm
        if self.pos_norm:
            self.norm = nn.LayerNorm(num_pos_feats)

    def init_weights(self):
        """Initialize the learnable weights."""
        uniform_init(self.row_embed_l1)
        uniform_init(self.col_embed_l1)
        uniform_init(self.z_embed_l1)
        uniform_init(self.row_embed_l2)
        uniform_init(self.col_embed_l2)
        uniform_init(self.z_embed_l2)
        uniform_init(self.row_embed_l3)
        uniform_init(self.col_embed_l3)
        uniform_init(self.z_embed_l3)

    def create_embed_level(self, mask, level):
        h = self.occ_h[level]
        w = self.occ_w[level]
        z = self.occ_z[level]
        x = torch.arange(h, device=mask.device)
        y = torch.arange(w, device=mask.device)
        zz = torch.arange(z, device=mask.device)
        
        if level == 0:
            x_embed = self.row_embed_l1(x)
            y_embed = self.col_embed_l1(y)
            zz_embed = self.z_embed_l1(zz)
            pos3d = x_embed.view(h,1,1,self.num_feats).expand(h,w,z,self.num_feats) + y_embed.view(1,w,1,self.num_feats).expand(h,w,z,self.num_feats) + zz_embed.view(1,1,z,self.num_feats).expand(h,w,z,self.num_feats)
            pos3d = pos3d.reshape(-1, self.num_feats)
            return pos3d[mask]
        elif level == 1:
            x_embed = self.row_embed_l2(x)
            y_embed = self.col_embed_l2(y)
            zz_embed = self.z_embed_l2(zz)
            pos3d = x_embed.view(h,1,1,self.num_feats).expand(h,w,z,self.num_feats) + y_embed.view(1,w,1,self.num_feats).expand(h,w,z,self.num_feats) + zz_embed.view(1,1,z,self.num_feats).expand(h,w,z,self.num_feats)
            pos3d = pos3d.reshape(-1, self.num_feats)
            return pos3d[mask]
        elif level == 2:
            x_embed = self.row_embed_l3(x)
            y_embed = self.col_embed_l3(y)
            zz_embed = self.z_embed_l3(zz)
            pos3d = x_embed.view(h,1,1,self.num_feats).expand(h,w,z,self.num_feats) + y_embed.view(1,w,1,self.num_feats).expand(h,w,z,self.num_feats) + zz_embed.view(1,1,z,self.num_feats).expand(h,w,z,self.num_feats)
            pos3d = pos3d.reshape(-1, self.num_feats)
            return pos3d[mask]
        else:
            assert False
    
    def forward(self, mask):
        pos_embed_list = []
        for i in range(3):
            pos_embed_list.append(self.create_embed_level(mask[i].reshape(-1), int(i)))
        
        pos_embed = torch.cat((pos_embed_list[0], pos_embed_list[1], pos_embed_list[2]), dim=0).cuda()

        if self.pos_norm:
            pos_embed = self.norm(pos_embed)

        return pos_embed