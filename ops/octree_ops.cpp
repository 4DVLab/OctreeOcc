#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA前向函数声明
void octree_mask_l1_to_l2_forward_cuda(
    const at::Tensor& octree_l1, 
    at::Tensor& mask_l2);

void octree_mask_l2_to_l3_forward_cuda(
    const at::Tensor& octree_l2, 
    at::Tensor& mask_l3);

// Python绑定函数
void octree_mask_l1_to_l2_forward(
    const at::Tensor& octree_l1, 
    at::Tensor& mask_l2) {
    
    // 确保输入在CUDA上
    TORCH_CHECK(octree_l1.is_cuda(), "octree_l1 must be a CUDA tensor");
    TORCH_CHECK(mask_l2.is_cuda(), "mask_l2 must be a CUDA tensor");
    
    // 确保输入是布尔类型
    TORCH_CHECK(octree_l1.scalar_type() == at::ScalarType::Bool, "octree_l1 must be bool tensor");
    TORCH_CHECK(mask_l2.scalar_type() == at::ScalarType::Bool, "mask_l2 must be bool tensor");
    
    // 确保维度正确
    TORCH_CHECK(octree_l1.dim() == 4, "octree_l1 must be a 4D tensor");
    TORCH_CHECK(mask_l2.dim() == 4, "mask_l2 must be a 4D tensor");
    
    // 确保形状正确
    int B = octree_l1.size(0);
    int H = octree_l1.size(1);
    int W = octree_l1.size(2);
    int D = octree_l1.size(3);
    
    TORCH_CHECK(mask_l2.size(0) == B, "batch size mismatch");
    TORCH_CHECK(mask_l2.size(1) == H*2, "height mismatch");
    TORCH_CHECK(mask_l2.size(2) == W*2, "width mismatch");
    TORCH_CHECK(mask_l2.size(3) == D*2, "depth mismatch");
    
    // 调用CUDA实现
    const at::cuda::OptionalCUDAGuard device_guard(device_of(octree_l1));
    octree_mask_l1_to_l2_forward_cuda(octree_l1, mask_l2);
}

void octree_mask_l2_to_l3_forward(
    const at::Tensor& octree_l2, 
    at::Tensor& mask_l3) {
    
    // 确保输入在CUDA上
    TORCH_CHECK(octree_l2.is_cuda(), "octree_l2 must be a CUDA tensor");
    TORCH_CHECK(mask_l3.is_cuda(), "mask_l3 must be a CUDA tensor");
    
    // 确保输入是布尔类型
    TORCH_CHECK(octree_l2.scalar_type() == at::ScalarType::Bool, "octree_l2 must be bool tensor");
    TORCH_CHECK(mask_l3.scalar_type() == at::ScalarType::Bool, "mask_l3 must be bool tensor");
    
    // 确保维度正确
    TORCH_CHECK(octree_l2.dim() == 4, "octree_l2 must be a 4D tensor");
    TORCH_CHECK(mask_l3.dim() == 4, "mask_l3 must be a 4D tensor");
    
    // 确保形状正确
    int B = octree_l2.size(0);
    int H = octree_l2.size(1);
    int W = octree_l2.size(2);
    int D = octree_l2.size(3);
    
    TORCH_CHECK(mask_l3.size(0) == B, "batch size mismatch");
    TORCH_CHECK(mask_l3.size(1) == H*2, "height mismatch");
    TORCH_CHECK(mask_l3.size(2) == W*2, "width mismatch");
    TORCH_CHECK(mask_l3.size(3) == D*2, "depth mismatch");
    
    // 调用CUDA实现
    const at::cuda::OptionalCUDAGuard device_guard(device_of(octree_l2));
    octree_mask_l2_to_l3_forward_cuda(octree_l2, mask_l3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("octree_mask_l1_to_l2_forward", &octree_mask_l1_to_l2_forward, "Octree mask L1 to L2 forward");
    m.def("octree_mask_l2_to_l3_forward", &octree_mask_l2_to_l3_forward, "Octree mask L2 to L3 forward");
}