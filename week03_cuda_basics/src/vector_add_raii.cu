#include<cmath>
#include<cuda_runtime.h>
#include<vector>
#include "../include/cuda_utils.cuh"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    // 1. Host 数据
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // 2. Device 数据 (RAII 自动管理显存)
    DeviceBuffer<float> d_a(N), d_b(N), d_c(N);

    // 3. 数据拷贝 to Device
    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 4. Kernel 调用
    const int block = 256;
    vector_add_kernel<<<(N + block -1) / block, block>>>(d_a.data(), d_b.data(), d_c.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 数据拷贝回 HOST
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.data(), bytes, cudaMemcpyDeviceToHost));

    // 6. 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    printf("RAII Vector Add: %s\n", correct ? "Correct" : "Incorrect");

    // 自动 cudaFree
    return 0;
}