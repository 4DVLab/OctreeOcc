#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// 常量定义在设备端，避免频繁内存传输
__constant__ int CHILD_OFFSETS[8][3] = {
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
    {1, 1, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}
};

// 高效的CUDA内核，一次性处理所有子节点
__global__ void octree_mask_kernel(
    const bool* __restrict__ input_mask,
    bool* __restrict__ output_mask,
    const int B, const int H, const int W, const int D,
    const int out_H, const int out_W, const int out_D) {
    
    // 计算当前线程对应的索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * H * W * D;
    
    if (index >= total_elements) return;
    
    // 计算input中的4D索引
    const int d = index % D;
    const int w = (index / D) % W;
    const int h = (index / (D * W)) % H;
    const int b = index / (D * W * H);
    
    // 获取原始值
    const int input_idx = ((b * H + h) * W + w) * D + d;
    const bool is_active = input_mask[input_idx];
    
    // 如果当前体素是活动的，设置所有8个子节点
    if (is_active) {
        // 计算输出中的基础索引
        const int h_out = h * 2;
        const int w_out = w * 2;
        const int d_out = d * 2;
        
        // 一次性计算8个子节点的索引并设置值
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_h = h_out + CHILD_OFFSETS[i][0];
            const int out_w = w_out + CHILD_OFFSETS[i][1];
            const int out_d = d_out + CHILD_OFFSETS[i][2];
            
            // 计算输出索引
            const int output_idx = ((b * out_H + out_h) * out_W + out_w) * out_D + out_d;
            output_mask[output_idx] = true;
        }
    }
}

void octree_mask_l1_to_l2_forward_cuda(
    const at::Tensor& octree_l1, 
    at::Tensor& mask_l2) {
    
    // 获取张量大小
    const int B = octree_l1.size(0);
    const int H = octree_l1.size(1);
    const int W = octree_l1.size(2);
    const int D = octree_l1.size(3);
    
    // 输出尺寸
    const int out_H = H * 2;
    const int out_W = W * 2;
    const int out_D = D * 2;
    
    const int total_elements = B * H * W * D;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // 选择最佳的流以提高性能
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 启动内核
    octree_mask_kernel<<<blocks, threads, 0, stream>>>(
        octree_l1.data_ptr<bool>(),
        mask_l2.data_ptr<bool>(),
        B, H, W, D,
        out_H, out_W, out_D
    );
}

void octree_mask_l2_to_l3_forward_cuda(
    const at::Tensor& octree_l2, 
    at::Tensor& mask_l3) {
    
    // 获取张量大小
    const int B = octree_l2.size(0);
    const int H = octree_l2.size(1);
    const int W = octree_l2.size(2);
    const int D = octree_l2.size(3);
    
    // 输出尺寸
    const int out_H = H * 2;
    const int out_W = W * 2;
    const int out_D = D * 2;
    
    const int total_elements = B * H * W * D;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // 选择最佳的流以提高性能
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 启动内核
    octree_mask_kernel<<<blocks, threads, 0, stream>>>(
        octree_l2.data_ptr<bool>(),
        mask_l3.data_ptr<bool>(),
        B, H, W, D,
        out_H, out_W, out_D
    );
} 