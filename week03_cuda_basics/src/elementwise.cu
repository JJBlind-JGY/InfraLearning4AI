// elementwise.cu
// Experiment 2: Elementwise Operations
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<cuda_runtime.h>

// 定义CHECK报错函数
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error : %s at (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); std::exit(EXIT_FAILURE);}} while (0)

// unfused affine relu kernel
// 数据流图
// x in HBM
// ↓
// Kernel 1
// ↓
// read x
// compute affine
// write tmp to HBM

// tmp in HBM
// ↓
// Kernel 2
// ↓
// read tmp
// compute ReLU
// write y to HBM
__global__ void affine_kernel(const float* x, float* tmp, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = alpha * x[i] + beta;        // 背景知识：这种“乘加”操作在 AI 中叫 Affine Transformation，是全连接层（Linear Layer）的基础
}

__global__ void relu_kernel(const float* tmp, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tmp[i] > 0.0f ? tmp[i] : 0.0f;
}


// fused affine relu kernel
// 数据流图         减少了 1 个 Kernel        Kernel Launch + Intermediate Memory Traffic
// x in HBM
// ↓
// Kernel 1
// ↓
// read x
// compute affine
// write y to HBM
__global__ void fused_affine_relu(const float* x, float* y, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = alpha * x[i] + beta;      // 理想情况下应该放在Register、Shared Memory中，而不是HBM或者是DRAM中，这就是融合计算kernel的好处
        y[i] = value > 0.0f ? value : 0.0f;     // 核心原理：value 被编译器优先分配在 寄存器（Register） 中，寄存器是 GPU 最快的存储（比 HBM 快约 100 倍）。这里没有额外的显存写入和读取
    }
}


int main() {
    const int N = 1 << 24;      // 16MB 数据量
    const float alpha = 1.5f, beta = -0.5f;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_x(N), h_y(N);       // 只分配了 x 和 y，没有分配 tmp（因为 fused 不需要）

    for (int i = 0; i < N; i++) h_x[i] = static_cast<float>(i % 1000) / 1000.0f - 0.5f;

    float * d_x = nullptr, *d_tmp = nullptr, *d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));      // 注意：这里虽然分配了 tmp，但 fused 没用到
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;     // N=16,777,216, block=256 → grid=65,536

    fused_affine_relu<<<grid, block>>>(d_x, d_y, alpha, beta, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        float value = alpha * h_x[i] + beta;
        float expected = value > 0.0f ? value : 0.0f;
        if (std::fabs(h_y[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }

    printf("Elementwise correctness: %s\n", correct ? "Pass" : "Fail");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_y));
    return correct ? 0 : 1;
}




// 为什么要学“融合”（Fusion）？
// 如果写成 Unfused（两步走）：
// HBM 读取 x (64 MB)
//        ↓ Kernel 1 算仿射
// HBM 写入 tmp (64 MB)   ← 这 64 MB 是多余的！
// HBM 读取 tmp (64 MB)   ← 这 64 MB 又是多余的！
//        ↓ Kernel 2 算 ReLU
// HBM 写入 y (64 MB)
// 总 HBM 流量：读 x (64) + 写 tmp (64) + 读 tmp (64) + 写 y (64) = 256 MB

// 如果写成 Fused（一步走）：
// HBM 读取 x (64 MB)
//        ↓ Kernel 内部：value 在寄存器中（不读写 HBM）
// HBM 写入 y (64 MB)
// 总 HBM 流量：读 x (64) + 写 y (64) = 128 MB

// 整整减少了一半的显存流量！ 对于 Memory-Bound 算子，这往往意味着 近乎翻倍的性能提升。这就是 FlashAttention 思想在微观尺度上的体现：让中间数据尽量留在 On-chip（寄存器/共享内存），别往 HBM 里倒腾




// 为什么 float value 能放在寄存器里？
// value 是一个局部变量，声明在 Kernel 函数体内。
// 对于 GPU 编译器（nvcc）来说，只要寄存器足够，它会自动把局部变量映射到寄存器，而不是显存。
// 只有当寄存器不够用（Register Pressure 过高）时，编译器才会把多余的变量“溢出（Spill）”到局部内存（Local Memory，实际上还是走显存），导致性能暴跌。在我们的例子里，一个线程只用 1 个 float 变量（4 字节），远低于寄存器的配额（通常每线程最多 255 个寄存器），所以 100% 在寄存器中，没有溢出。