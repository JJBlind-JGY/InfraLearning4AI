// vector_add.cu
// Experiment 1: Vector Add

// 数学任务： C_i = A_i + B_i
// 其中，A_i, B_i, C_i 分别表示向量 A, B, C 的第 i 个元素  索引范围为 [0, n-1]

// 一般情况下，CPU 的写法是这样的形式
// for (int i = 0; i< N; i++) c[i] = a[i] + b[i];
// CUDA 的本质想法就是不希望一个 CPU Thread 循环N次来实现，而是创建多个CUDA Threads，每一个CUDA Thread 处理一个元素的计算
// Thread 0 → C[0]
// Thread 1 → C[1]
// Thread 2 → C[2]
// ...

// 目标：
// 在 GPU 上做大规模向量加法：C[i] = A[i] + B[i]，一共 N = 1 << 20 = 1,048,576 个元素。
// 先在 CPU（Host）内存里准备好两个数组，然后拷贝到 GPU（Device）显存，让 GPU 成千上万个线程同时做加法，最后把结果拷回 CPU 验证。


// 流程：
// CPU
// │
// ├── std::vector h_a
// ├── std::vector h_b
// └── std::vector h_c
//      │
//      │ cudaMalloc
//      ↓
// GPU HBM
// │
// ├── d_a
// ├── d_b
// └── d_c

// Host → Device
// cudaMemcpy
//      ↓

// vector_add<<<grid,block>>>
//      ↓

// GPU Threads
//      ↓
// read d_a
// read d_b
// add
// write d_c

//      ↓
// Device → Host
// cudaMemcpy
//      ↓
// 验证结果




#include<cmath>     // std::fabs 计算绝对值
#include<cstdio>    // printf 
#include<cstdlib>   // std::exit
#include<vector>    // std::vector 容义动态数组
#include<cuda_runtime.h> //  cuda 运行时API ：cudaMalloc, cudaMemcpy, cudaFreeLastError, cudaDeviceSynchronize 等 和 内置变量 blockIdx, blockDim, threadIdx, blockDim


// 定义一个宏 【在编译前，预处理器会把代码中所有 CUDA_CHECK( something ) 替换成后续的内容】 
// 目的：简化错误检查。CUDA 的每个 API 函数（如 cudaMalloc）都会返回一个错误码（cudaError_t），成功时返回 cudaSuccess。如果不检查，一旦出错程序还会继续跑，后面会崩溃得莫名其妙
// do { ... } while (0) 技巧：这是 C/C++ 宏的标准写法，让宏像一个单独的语句一样工作，不会受 if 等语法干扰。
// cudaGetErrorString(err)：把错误码转成人类可读的字符串，比如 "out of memory"。
// __FILE__ 和 __LINE__：预定义宏，分别表示当前文件名和行号，方便定位错误
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE);}} while (0)


// __global__：告诉编译器，这个函数是 Kernel，即 在 GPU 上执行，从 CPU 端调用（通过 <<<>>>）
// 参数：都是指针和整数 a、b、c 是 指向 GPU 内存的指针，它们保存的是 GPU 显存地址，不是 CPU 内存地址；n 是普通 int，直接传值（硬件会把值复制到每个线程）
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;      // global thread id = block id * block size + local thread id
    if (i < n) {        // 检查thread id是否越界 （最后一个block的thread id可能会超出范围）
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;      // 位操作：1 << 20 表示 2^20 = 1,048,576 个元素   --> 这么写是为了保持数值是 2 的幂，方便 Block/Grid 划分

    // size_t：无符号整数类型，专门用来表示大小（内存字节数），保证能容纳大尺寸
    // static_cast<size_t>(N) 先把 N 转成 size_t，再乘以 sizeof(float)（4 字节），得到总共需要的字节数：1,048,576 × 4 = 4,194,304 字节（约 4 MB）
    const size_t bytes = static_cast<size_t>(N) * sizeof(float); 

    std::vector<float> h_a(N), h_b(N), h_c(N);      // 三个动态数组，分别表示向量 A, B, C 的元素， 每一个N个元素占4字节
    // std::vector 是 C++ 的动态数组，内存分配在 CPU 的堆（Heap） 上
    // 命名 h_ 前缀代表 Host（CPU 端），方便区分 GPU 端的 d_（Device）指针
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);      // 初始化向量 A 的第 i 个元素为 i
        h_b[i] = static_cast<float>(2 * i);   // 初始化向量 B 的第 i 个元素为 2 * i
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;       // 这三个指针的类型是 float*，但它们现在指向空（nullptr）。它们将被用来保存 GPU 显存地址。
    // cudaMalloc：在 GPU 显存中分配 bytes 个字节，并 把这块显存的起始地址存入 d_a。
    // 为什么传 &d_a 而不是 d_a？
    // d_a 本身是一个变量，它的类型是 float*（一个地址值）
    // cudaMalloc 需要 修改 d_a 的值（把新分配的地址写进去）
    // 为了修改外部变量，C/C++ 必须传递 变量的地址，即 &d_a，类型是 float**（指针的指针）
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));



    // cudaMemcpy：从 CPU（Host）内存拷贝 bytes 个字节到 GPU（Device）显存 d_a。  参数依次是：目标地址、源地址、字节数、拷贝方向。
    // 方向：
    // cudaMemcpyHostToDevice：从 CPU 内存拷贝到 GPU 显存
    // cudaMemcpyDeviceToHost：从 GPU 显存拷贝到 CPU 内存
    // cudaMemcpyDeviceToDevice：从 GPU 显存拷贝到 GPU 显存
    // cudaMemcpyHostToHost：从 CPU 内存拷贝到 CPU 内存

    // h_a.data() 是 std::vector 的方法，返回指向内部数组起始位置的指针（CPU 地址）。
    // d_a 是 GPU 显存地址。
    // 作用：把 CPU 的 h_a 和 h_b 数组完整复制到 GPU 的 d_a 和 d_b 中。此时，GPU 有了两份输入数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;     // 每个 Block 包含 256 个 Threads。这是 CUDA 编程的常见起始值（因为 256 是 32 的倍数，正好 8 个 Warps）
    // grid_size：向上取整 的经典公式。
    // 如果 N = 1000，block_size = 256，那么 grid_size = (1000 + 255) / 256 = 1255 / 256 = 4（整数除法）。
    // 这意味着启动 4 个 Block，总共 4 × 256 = 1024 个 Threads，但实际只有 1000 个元素。多出来的 24 个线程靠 if (i < n) 来忽略
    const int grid_size = (N + block_size - 1) / block_size;




    // <<<grid_size, block_size>>>：启动参数，告诉 GPU 启动 grid_size 个 Block，每个 Block 有 block_size 个 Thread。
    // 参数传递：d_a、d_b、d_c（都是 GPU 地址）、N（普通整数）被传给 Kernel 函数。
    // 此时，GPU 上启动了多少个线程？ grid_size × block_size 个。如果 N 是 1<<20，block_size=256，那么 grid_size = 4096，总线程数 = 4096 × 256 = 1,048,576，正好等于 N。
    // 异步特性：CPU 发出这个命令后，立即返回，不会等待 GPU 执行完毕。GPU 会在后台慢慢执行
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());     // 检查 Kernel 启动时是否发生了错误（例如配置不合理）
    CUDA_CHECK(cudaDeviceSynchronize());     // 强制 CPU 阻塞等待，直到 GPU 上所有先前发出的命令全部执行完毕。如果没有这行，CPU 可能直接进行下一步 cudaMemcpy（从 GPU 拷回结果），而 GPU 还没算完，拷回来的就是垃圾数据
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));     // 从 GPU 显存 d_c 拷贝 bytes 个字节到 CPU 内存 h_c 数组。作用：把 GPU 的计算结果拷贝到 CPU 内存，方便后续检查


    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            printf("Mismatch at %d: got=%f expected=%f\n", i, h_c[i], expected);
            break;
        }
    }

    printf("VectorAdd Correctness: %s\n", correct ? "PASS" : "FAIL");
    printf("Grid size : %d\n", grid_size);
    printf("Block size: %d\n", block_size);

    // GPU 显存不会自动释放，必须手动 cudaFree。如果你忘记释放，程序结束时会自动释放，但长时间运行的程序会内存泄漏，最终耗尽显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return correct ? 0 : 1;
}




