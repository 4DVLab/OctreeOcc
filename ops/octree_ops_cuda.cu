#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

// 定义线程块大小
const int THREADS_PER_BLOCK = 256;

// CUDA kernel实现
__global__ void octree_mask_l1_to_l2_kernel(
    const bool* __restrict__ octree_l1,
    bool* __restrict__ mask_l2,
    const int B,
    const int H,
    const int W,
    const int D) {
    
    // 计算全局索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * H * W * D;
    
    if (index >= total_elements) return;
    
    // 计算input中的4D索引
    const int d = index % D;
    const int w = (index / D) % W;
    const int h = (index / (D * W)) % H;
    const int b = index / (D * W * H);
    
    // 取得原始值
    const int input_idx = ((b * H + h) * W + w) * D + d;
    const bool value = octree_l1[input_idx];
    
    // 如果值为true，则计算在输出中的8个子节点位置并设置
    if (value) {
        // 计算在输出中的基础索引
        const int h_out = h * 2;
        const int w_out = w * 2;
        const int d_out = d * 2;
        
        // 8个子节点的偏移量
        const int offsets[8][3] = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
            {1, 1, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}
        };
        
        // 对8个子节点赋值
        for (int i = 0; i < 8; ++i) {
            const int h_offset = offsets[i][0];
            const int w_offset = offsets[i][1];
            const int d_offset = offsets[i][2];
            
            const int output_idx = ((b * (H*2) + (h_out+h_offset)) * (W*2) + (w_out+w_offset)) * (D*2) + (d_out+d_offset);
            mask_l2[output_idx] = true;
        }
    }
}

// CUDA kernel实现 - 同样逻辑适用于L2到L3
__global__ void octree_mask_l2_to_l3_kernel(
    const bool* __restrict__ octree_l2,
    bool* __restrict__ mask_l3,
    const int B,
    const int H,
    const int W,
    const int D) {
    
    // 计算全局索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * H * W * D;
    
    if (index >= total_elements) return;
    
    // 计算input中的4D索引
    const int d = index % D;
    const int w = (index / D) % W;
    const int h = (index / (D * W)) % H;
    const int b = index / (D * W * H);
    
    // 取得原始值
    const int input_idx = ((b * H + h) * W + w) * D + d;
    const bool value = octree_l2[input_idx];
    
    // 如果值为true，则计算在输出中的8个子节点位置并设置
    if (value) {
        // 计算在输出中的基础索引
        const int h_out = h * 2;
        const int w_out = w * 2;
        const int d_out = d * 2;
        
        // 8个子节点的偏移量
        const int offsets[8][3] = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
            {1, 1, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}
        };
        
        // 对8个子节点赋值
        for (int i = 0; i < 8; ++i) {
            const int h_offset = offsets[i][0];
            const int w_offset = offsets[i][1];
            const int d_offset = offsets[i][2];
            
            const int output_idx = ((b * (H*2) + (h_out+h_offset)) * (W*2) + (w_out+w_offset)) * (D*2) + (d_out+d_offset);
            mask_l3[output_idx] = true;
        }
    }
}

// CUDA接口函数
void octree_mask_l1_to_l2_forward_cuda(
    const at::Tensor& octree_l1, 
    at::Tensor& mask_l2) {
    
    // 获取张量大小
    const int B = octree_l1.size(0);
    const int H = octree_l1.size(1);
    const int W = octree_l1.size(2);
    const int D = octree_l1.size(3);
    const int total_elements = B * H * W * D;
    
    // 计算grid大小
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // 调用CUDA kernel
    octree_mask_l1_to_l2_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        octree_l1.data_ptr<bool>(),
        mask_l2.data_ptr<bool>(),
        B, H, W, D
    );
    
    // 同步CUDA流
    cudaDeviceSynchronize();
}

// CUDA接口函数
void octree_mask_l2_to_l3_forward_cuda(
    const at::Tensor& octree_l2, 
    at::Tensor& mask_l3) {
    
    // 获取张量大小
    const int B = octree_l2.size(0);
    const int H = octree_l2.size(1);
    const int W = octree_l2.size(2);
    const int D = octree_l2.size(3);
    const int total_elements = B * H * W * D;
    
    // 计算grid大小
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // 调用CUDA kernel
    octree_mask_l2_to_l3_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        octree_l2.data_ptr<bool>(),
        mask_l3.data_ptr<bool>(),
        B, H, W, D
    );
    
    // 同步CUDA流
    cudaDeviceSynchronize();
}