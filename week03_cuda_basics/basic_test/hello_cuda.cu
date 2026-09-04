#include <cstdio>
#include <cuda_runtime.h> // 引入 cuda 运行时 API （cudaMalloc、cudaMemcpy等均在此头文件中）

// 由于 cuda 版本 和 cl.exe 的版本不一致，导致编译会出现报错，所以需要配置运行参数
// 按照如下指令运行：nvcc -allow-unsupported-compiler .\hello_cuda.cu -o hello_cuda

// 对于一维的kernel，应该是满足 kernel<<<num_blocks, threads_per_block>>>();
__global__ void hello_cuda() {
    // __global__ 是一个cuda关键字，表示这个函数 Kernel(核函数) -- 在GPU上执行，但是从CPU端调用
    // CUDA Kernel 函数中可以使用 printf(仅限计算能力 >= 2.0 的 GPU)，但是极度影响性能，仅用来调试
    // blockIdx.x / threadIdx.x 是内置变量，分别表示当前 Block 在 Grid 中的编号 和 当前 Thread 在 Block 中的编号（一维状况）
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // 理解 <<<2, 4>>>：Grid 有 2 个 Block，每个 Block 有 4 个 Thread，共 8 个并行线程。 
    // <<<Grid Size, Block Size>>> 表示 Grid 中的 Block 数量和每个 Block 中的 Thread 数量。
    hello_cuda<<<2, 4>>>();  // 2 个 block, 每个 4 个 thread
    cudaDeviceSynchronize();     // 等待所有 thread 完成
    return 0;

    // hello_cuda<<<2, 4>>>()	这是 Kernel Launch 语法。<<<A, B>>> 表示：启动 A 个 Block，每个 Block 有 B 个 Thread。这里就是 2 个 Block，每个 4 个 Thread。
    // cudaDeviceSynchronize()	强制 CPU 等待 GPU 完成所有已发射的 Kernel 再继续。因为 Kernel 是 异步 启动的，不加这个，CPU 可能直接 return 0 退出程序，GPU 还没开始打印。

}



// ===========================================CUDA 执行模型拆解（硬件视角）==========================================
// <<<2, 4>>> 在硬件上发生了什么？

// GPU Grid（一次启动的总任务）
// │
// ├── Block 0（被调度到某个 SM，比如 SM 0）
// │   ├── Thread 0 → 打印 "Hello from block 0, thread 0"
// │   ├── Thread 1 → 打印 "Hello from block 0, thread 1"
// │   ├── Thread 2 → 打印 "Hello from block 0, thread 2"
// │   └── Thread 3 → 打印 "Hello from block 0, thread 3"
// │
// └── Block 1（被调度到另一个 SM，或者同一个 SM 的不同时间片）
//     ├── Thread 0 → ...
//     ├── Thread 1
//     ├── Thread 2
//     └── Thread 3

// 关键点：Block 0 和 Block 1 可能同时执行（如果 GPU 有至少 2 个空闲 SM），也可能 先后执行（如果 SM 资源紧张）。你无法控制顺序，只能假设它们独立。
// Warp 登场：虽然 Block 里只有 4 个 Thread，但 NVIDIA GPU 的硬件调度单位是 Warp（32 个 Thread）。所以这 4 个 Thread 会被凑成一个 不满的 Warp（只有 4 个活跃 Lane，其余 28 个处于非活跃状态）。这会造成 资源浪费，但初学者无需担心。

// ===========================================内存模型拆解==========================================
// 这个例子里几乎没有内存操作，但注意：

// blockIdx.x 和 threadIdx.x 是 内置寄存器变量（存放在 Thread 私有的 Register 中），读取极快。

// printf 的输出会通过 GPU 的特殊管道传回 主机Host 的 stdout，不是直接写 HBM，但涉及同步开销。

// ===========================================设计原理（为什么这样设计？）==========================================
// 为什么需要 <<<>>> 语法？
// 因为 CPU 是单线程启动者，GPU 是海量并行执行者。CPU 必须告诉 GPU：“我要启动多少并行任务”。<<<grid, block>>> 就是这座桥梁。如果不指定，GPU 不知道开多少个线程。

// 为什么需要 cudaDeviceSynchronize()？
// 因为 CPU 和 GPU 是异步协作的。CPU 发出启动信号后，立即继续执行下一行（就像老板给工人下完命令后继续看下一份文件，不等工人干完）。如果不等待，程序可能直接结束，工人（GPU）还没开始干活。
