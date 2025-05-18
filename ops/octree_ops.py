import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import os

# 动态加载CUDA扩展
current_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name='octree_ops',
    sources=[
        os.path.join(current_dir, 'octree_ops.cpp'),
        os.path.join(current_dir, 'octree_ops_cuda.cu')
    ],
    extra_cflags=['-O3'],
    verbose=True
)

class OctreeMaskL1ToL2Function(Function):
    @staticmethod
    def forward(ctx, octree_l1):
        """从L1级别生成L2级别的八叉树掩码
        
        Args:
            octree_l1 (torch.Tensor): [B, H, W, D] 布尔值掩码
            
        Returns:
            torch.Tensor: [B, 2*H, 2*W, 2*D] 布尔值掩码
        """
        # 确保输入是布尔值
        octree_l1 = octree_l1.bool()
        
        # 创建输出张量
        B, H, W, D = octree_l1.shape
        mask_l2 = torch.zeros(B, H*2, W*2, D*2, dtype=torch.bool, device=octree_l1.device)
        
        # 调用CUDA操作
        _ext.octree_mask_l1_to_l2_forward(octree_l1, mask_l2)
        
        return mask_l2
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 该操作不需要梯度计算
        return None

class OctreeMaskL2ToL3Function(Function):
    @staticmethod
    def forward(ctx, octree_l2):
        """从L2级别生成L3级别的八叉树掩码
        
        Args:
            octree_l2 (torch.Tensor): [B, H, W, D] 布尔值掩码
            
        Returns:
            torch.Tensor: [B, 2*H, 2*W, 2*D] 布尔值掩码
        """
        # 确保输入是布尔值
        octree_l2 = octree_l2.bool()
        
        # 创建输出张量
        B, H, W, D = octree_l2.shape
        mask_l3 = torch.zeros(B, H*2, W*2, D*2, dtype=torch.bool, device=octree_l2.device)
        
        # 调用CUDA操作
        _ext.octree_mask_l2_to_l3_forward(octree_l2, mask_l3)
        
        return mask_l3
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 该操作不需要梯度计算
        return None

# 对外暴露的API函数
def create_octree_mask_l1_to_l2(octree_l1):
    return OctreeMaskL1ToL2Function.apply(octree_l1)

def create_octree_mask_l2_to_l3(octree_l2):
    return OctreeMaskL2ToL3Function.apply(octree_l2)