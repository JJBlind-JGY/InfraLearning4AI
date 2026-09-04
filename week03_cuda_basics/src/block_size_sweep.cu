// block_size_sweep.cu
// Experiment 3: Block Size Sweep

// 做一件事：探索 Block Size 对 VectorAdd 性能的影响
// 1. 固定输入数组大小 N = 1 << 26（约 6700 万个元素，每数组 256 MB）
// 2. 遍历 6 种 Block Size：32, 64, 128, 256, 512, 1024
// 3. 对每一种，运行 100 次 VectorAdd Kernel，取平均耗时
// 4. 根据耗时推算出“有效带宽”（GB/s），并输出到终端和 CSV 文件


#include<cstdio>
#include<cstdlib>
#include<fstream>       // 文件读写  C++ 标准库的文件操作，用于把结果写入 block_size_results.csv，方便后续用 Excel 或 Python 画图
#include<vector>
#include<cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s at (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); std::exit(EXIT_FAILURE);}} while(0)

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

float benchmark(const float* a, const float* b, float* c, int n, int block_size, int warmups, int repeats) {
    int grid_size = (n + block_size - 1) / block_size;


    // 为什么需要预热？
    // -- GPU 频率提升：现代 GPU 有动态频率管理（类似 CPU 的睿频），刚启动时频率可能较低，跑几轮后频率才会稳定到最高值
    // -- 上下文加载：Kernel 代码第一次执行时，可能需要从显存加载到指令缓存，或者建立页表映射
    // -- 消除冷启动噪音：如果不预热，第一次执行的时间可能异常偏大，污染你的测量结果
    // 同步：预热后立即同步，确保所有预热 Kernel 执行完毕，然后再开始正式计时，防止前后干扰
    for (int i = 0; i < warmups; i++) {
        vector_add<<<grid_size, block_size>>>(a, b, c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());



    // cudaEvent_t：CUDA 事件类型，它是一个时间戳标记点，记录在 GPU 的命令流（Stream）中
    // cudaEventCreate：在 GPU 上分配一个事件对象。

    // 为什么不用 std::chrono（CPU 时钟）？
    // -- std::chrono 测量的是 CPU 代码执行耗时，包括 Kernel 启动的开销和驱动调度的延迟
    // -- cudaEvent 记录的是 GPU 端实际执行命令的时间线，它更准确，且不受 CPU 调度干扰
    // -- 此外，Kernel 是异步启动的（CPU 发完指令就继续），若用 CPU 计时器，你测量的很可能只是 <<<>>> 启动的软件开销，而非真正的计算耗时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));     // 在 GPU 命令流中插入“开始”标记
    // 循环启动 repeats 次 Kernel。因为 GPU 是异步的，这些 Kernel 会排队执行，但 CPU 会立即返回并继续下一次循环（所以循环本身很快）
    for (int i = 0; i < repeats; i++) vector_add<<<grid_size, block_size>>>(a, b, c, n);
    CUDA_CHECK(cudaEventRecord(stop));      // 在 GPU 命令流中插入“结束”标记
    CUDA_CHECK(cudaEventSynchronize(stop)); // 阻塞 CPU，直到 GPU 执行到 stop 这个时间戳为止。此时，从 start 到 stop 之间的所有 Kernel 都已经执行完毕  // 等待确保所有命令都完成


    //cudaEventElapsedTime：计算两个事件之间的时间差，单位是毫秒（ms），结果填入 total_ms
    //cudaEventDestroy：销毁事件对象，释放 GPU 资源
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeats;      //  单次 Kernel 的平均耗时（毫秒），因为运行了 repeats 次，取平均可以消除偶然波动（比如操作系统中断）
}

int main() {
    const int N = 1 << 26;      // 6700 万个元素，每数组 256 MB
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);     // 每个元素 4 字节，所以总大小是 N * sizeof(float)


    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 本次实验只测带宽（吞吐量），不测正确性。
    // VectorAdd 的耗时主要由读写显存的总字节数决定，跟数值是 0 还是 123 没有关系（浮点加法器的延迟固定）
    // 所以用 cudaMemset 快速清零，省掉了 Host→Device 的传输时间（那些跟 Kernel 时间无关，只测 Kernel 执行时间）
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));          // 初始化数组 a 为 0
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));          // 初始化数组 b 为 0

    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};

    std::ofstream csv("block_size_results.csv");
    csv << "block_size,warps_per_block,time_ms,effective_bandwidth_gbs\n";

    printf("%10s %12s %12s %20s\n", "Block", "Warps", "Time(ms)", "Bandwidth(GB/s)");

    for (int block: block_sizes) {
        float ms = benchmark(d_a, d_b, d_c, N, block, 10, 1000);

        double seconds = ms / 1000.0;
        double transferred_bytes = 3.0 * static_cast<double>(bytes);                // A B read & C write --> 3 times
        double bandwidth_gbs = transferred_bytes / seconds / 1e9;       // 字节数除以时间，再除以 10^9，得到 GB/s（十进制，非 1024 进制）

        printf("%10d %12d %12.4f %20.2f\n", block, block/32, ms, bandwidth_gbs);
        csv << block << "," << block/32 << "," << ms << "," << bandwidth_gbs << "\n";
    }

    csv.close();

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}


// 细节 1：cudaMemset 清零会影响 Cache 行为吗？
// 清零后，显存页面的内容被初始化为 0，但不影响后续 Kernel 的 Cache 行为。第一次读这些页面时，数据从 HBM 搬运到 L2/SRAM；后续重复 Kernel 运行时，L2 可能缓存部分数据，导致第二次比第一次快。但我们在循环里重复跑 100 次，后 99 次可能会受益于 Cache，而第一次（warmup）也会填充 Cache。最终测得的带宽是包含 Cache 效应的有效带宽，这实际上是真实应用场景下的水平，所以没问题。

// 细节 2：为什么 block_sizes 都是 32 的倍数？
// 因为 NVIDIA GPU 的 Warp 大小固定为 32。如果 Block Size 不是 32 的倍数（比如 33），最后一个 Warp 只有 1 个活跃线程，其余 31 个线程被浪费掉（Divergence 和 Occupancy 都变差）。实际编程中几乎总是使用 32 的倍数。