# Week 03｜GPU Architecture & CUDA Basics 工程实践报告

> **主题：GPU Execution Model、CUDA VectorAdd、Elementwise Kernel、Kernel Fusion、Block Size Sweep、Effective Bandwidth、C++ 基础与 CPU Cache**
>
> **本周目标：**
>
> $$
> \boxed{
> \text{CUDA Code}
> \rightarrow
> \text{Grid / Block / Warp / Thread}
> \rightarrow
> \text{SM Execution}
> \rightarrow
> \text{Memory Access}
> \rightarrow
> \text{Benchmark}
> \rightarrow
> \text{Performance Explanation}
> }
> $$
>
> 本周不是为了“会写几个 CUDA Kernel”而完成实验，而是第一次建立：
>
> > **从 CUDA 源码出发，解释一个 Kernel 在 GPU 上到底由谁执行、数据从哪里来、Block Size 为什么影响性能、为什么 VectorAdd 主要受 Memory Bandwidth 限制，以及 Kernel Fusion 为什么可以减少 HBM Traffic。**

---

## 0. 实验内容总览

| 序号 | 实验名称 | 核心知识点 |
| :---: | :--- | :--- |
| 1 | CUDA 环境与第一个 Kernel | `__global__`, `<<<>>>`, `blockIdx`, `threadIdx` |
| 2 | CUDA VectorAdd | Grid/Block/Thread 映射，`cudaMalloc`, `cudaMemcpy` |
| 3 | Elementwise Kernel (Affine) | 逐元素运算，基本访存模式 |
| 4 | Affine + ReLU Kernel Fusion | 减少 HBM 中间流量，寄存器复用 |
| 5 | Block Size Sweep | Warp、Occupancy、调度粒度 |
| 6 | Effective Bandwidth Benchmark | 性能归一化测量，CUDA Event |
| 7 | Kernel 执行路径图 | 硬件流水线、SM、Warp 调度 |
| 8 | C++ 副轨：Pointer/Reference/RAII/STL | 内存安全，RAII 封装，现代 C++ 实践 |
| 9 | CPU Cache Locality 实验 | 行主序 vs 列主序，Cache 友好访问 |
| 10 | 成果整理 | 仓库、报告、简历证据 |

**本周核心问题链：**

```
Kernel 如何 Launch？
        ↓
Grid / Block / Thread 如何编号？
        ↓
Block 如何被分配给 SM？
        ↓
Thread 为什么按 Warp 执行？
        ↓
数据为什么从 HBM 读取？
        ↓
为什么连续访问更快？
        ↓
为什么 Block Size 会影响性能？
        ↓
为什么 Elementwise Fusion 会更快？
        ↓
如何正确 Benchmark 一个 GPU Kernel？
```

---

## 1. 从 Week 02 到 Week 03：知识衔接

在前两周中，我们已经建立了 **Arithmetic Intensity** 的概念：

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

并且理解了 LLM 推理中大量时间花在 **Memory Bandwidth Pressure** 上（Decode 阶段读取 Weight 和 KV Cache）。Week 03 开始真正从 **GPU 硬件执行层** 观察这些现象，并通过亲手编写 CUDA Kernel 来体会：

> 为什么 Memory Traffic 是性能的首要限制？  
> 如何通过调整执行配置和 Kernel 设计来改善？

因此，Week 03 是整个课程从 **“模型资源分析”** 到 **“硬件执行机制”** 的关键转折点。

---

## 2. CUDA Host / Device 基本模型

CUDA 程序同时运行在 **Host（CPU）** 和 **Device（GPU）** 上。

### Host（CPU）职责
- 控制程序流程
- 分配 Host 内存（`std::vector`、`malloc`）
- 准备输入数据
- 调用 CUDA Runtime API（`cudaMalloc`, `cudaMemcpy`）
- 启动 Kernel（`<<<>>>`）
- 同步与错误检查
- 验证结果

### Device（GPU）职责
- 执行 CUDA Kernel（`__global__` 函数）
- 大规模并行计算
- 访问 GPU Device Memory（Global、Shared、Register 等）

**典型 CUDA 程序执行流程：**

```
CPU Host
│
├── 1. 分配 Host 内存并填充数据
├── 2. cudaMalloc 分配 Device 内存
├── 3. cudaMemcpy Host → Device
├── 4. kernel<<<grid, block>>>(...)  // 异步启动
│         ↓
│        GPU 执行 Kernel
│         ↑
├── 5. cudaDeviceSynchronize()        // 等待 GPU 完成
├── 6. cudaMemcpy Device → Host
├── 7. Correctness Check
└── 8. cudaFree 释放 Device 内存
```

---

## 3. 第一个 CUDA Kernel：Hello World

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_cuda<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

- **`__global__`**：声明这是一个 Kernel（在 GPU 上执行，从 CPU 调用）。
- **`<<<2, 4>>>`**：启动 2 个 Block，每个 Block 4 个 Thread。
- **`blockIdx.x`**：当前 Block 在 Grid 中的一维索引。
- **`threadIdx.x`**：当前 Thread 在 Block 中的一维索引。

**执行结构：**

```
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── Thread 2
│   └── Thread 3
└── Block 1
    ├── Thread 0
    ├── Thread 1
    ├── Thread 2
    └── Thread 3
```

总线程数 = 2 × 4 = 8。每个线程打印自己的 Block 和 Thread ID（顺序不保证）。

---

## 4. `<<<grid, block>>>` 详解

语法：`kernel_name<<<grid_dim, block_dim>>>(args...)`

- **`grid_dim`**：Grid 中 Block 的数量（一维、二维或三维，这里为一维）。
- **`block_dim`**：每个 Block 中 Thread 的数量（同样支持多维）。

对于一维情况：

$$
\text{TotalThreads} = \text{grid\_dim} \times \text{block\_dim}
$$

这些 Threads 是逻辑上的并行单元，**并不等于** GPU 上物理核心的数量；硬件会将它们调度到有限的 SM 上分批执行。

---

## 5. 实验一：CUDA VectorAdd

### 数学任务
$$
C_i = A_i + B_i, \quad i = 0, 1, \ldots, N-1
$$

### CPU 串行实现
```cpp
for (int i = 0; i < N; ++i)
    c[i] = a[i] + b[i];
```

### CUDA 并行实现
```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

每个 Thread 负责一个元素，将循环展开为大规模并行。

### 完整的 `vector_add.cu`

```cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;   // 1,048,576
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;   // 向上取整

    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Correctness check
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            printf("Mismatch at %d: got %f expected %f\n", i, h_c[i], expected);
            break;
        }
    }
    printf("VectorAdd correctness: %s\n", correct ? "PASS" : "FAIL");
    printf("Grid size: %d, Block size: %d\n", grid_size, block_size);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return correct ? 0 : 1;
}
```

---

## 6. 核心索引公式

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

含义：将 Thread 的二维坐标（Block ID + Thread ID）映射为全局一维索引。

**示例**：若 `blockDim.x = 256`

| Block ID | Thread ID 范围 | 全局索引范围 |
| :---: | :---: | :---: |
| 0 | 0 ~ 255 | 0 ~ 255 |
| 1 | 0 ~ 255 | 256 ~ 511 |
| 2 | 0 ~ 255 | 512 ~ 767 |
| ... | ... | ... |

由此，每个元素被唯一一个 Thread 处理。

---

## 7. Grid Size 计算（向上取整）

```cpp
int grid_size = (N + block_size - 1) / block_size;
```

数学形式：$\lceil N / \text{block\_size} \rceil$

**示例**：$N=1000,\ \text{block\_size}=256$

$$
\text{grid\_size} = \left\lceil \frac{1000}{256} \right\rceil = 4
$$

实际启动 Threads = $4 \times 256 = 1024$，最后 24 个 Thread 无对应数据，由 `if (i < n)` 处理。

---

## 8. 为什么需要 `if (i < n)`？

当 $N$ 不是 `block_size` 的整数倍时，多余的 Thread 会产生越界访问。例如上面 1024 个 Thread 中，索引 1000~1023 对应的 `a[1000]`、`a[1001]` …… 越界。`if` 让这些 Thread 直接返回，保证程序安全。

---

## 9. CUDA 内存分配与指针含义

### Host 数据
```cpp
std::vector<float> h_a(N), h_b(N), h_c(N);
```

### Device 指针
```cpp
float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
```
这些指针变量存储在 CPU 内存中，但它们的值将来会被赋为 **GPU 显存地址**（Device Address）。

### `cudaMalloc(&d_a, bytes)`
- `d_a` 是一个 `float*`，用于保存 GPU 地址。
- `&d_a` 是 `float**`，即指针的地址。
- `cudaMalloc` 需要修改 `d_a` 的值（分配新地址），因此必须传 `&d_a`。

**类比**：如果你想让一个函数修改一个整型变量 `int x`，你必须传 `&x`。同理，修改指针变量也要传其地址。

---

## 10. Host ↔ Device 数据传输

```cpp
cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);  // H→D
cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);  // D→H
```

数据流向：

```
CPU DRAM (h_a)  --H2D-->  GPU HBM (d_a)
GPU HBM (d_c)   --D2H-->  CPU DRAM (h_c)
```

---

## 11. 验证正确性（Golden Reference）

使用 CPU 重新计算期望值，再与 GPU 结果比较，允许微小误差（`1e-5`）。

工程顺序：
1. 先保证 **Correctness**。
2. 再开始 **Benchmark**。
3. 最后进行 **Optimization**。

---

## 12. CUDA 错误检查宏

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)
```

- 包装 CUDA API 调用，自动检查返回值。
- 利用 `__FILE__` 和 `__LINE__` 精确定位错误。
- `do { ... } while (0)` 使得宏可以像普通语句一样使用（加分号）。

---

## 13. VectorAdd 数据路径全图

```
CPU
│
├── h_a, h_b, h_c
│
↓ cudaMalloc
│
GPU HBM
│
├── d_a, d_b, d_c
│
↓ cudaMemcpy Host→Device
│
GPU HBM (已填充 d_a, d_b)
│
↓ vector_add<<<grid, block>>>
│
GPU Threads 并行执行
│
├── Load d_a[i]
├── Load d_b[i]
├── Add
└── Store d_c[i]
│
↓ cudaDeviceSynchronize()
│
GPU HBM (d_c 已写入)
│
↓ cudaMemcpy Device→Host
│
CPU (h_c 已获得结果)
│
↓ Correctness Check
```

---

## 14. 为什么 VectorAdd 是 Memory-Bound？

每个 FP32 元素需完成：

| 操作 | 字节数 |
| :--- | :---: |
| 读取 `a[i]` | 4 |
| 读取 `b[i]` | 4 |
| 写入 `c[i]` | 4 |
| 加法（FLOP） | 1 |

总数据移动 = 12 Bytes，计算 = 1 FLOP，因此：

$$
\text{Arithmetic Intensity} \approx \frac{1}{12} \approx 0.083\ \text{FLOP/Byte}
$$

远低于 GPU 峰值 AI（通常 > 10），因此性能受限于 **显存带宽**，而非计算能力。

---

## 15. Coalesced Global Memory Access

在 VectorAdd 中，同一 Warp 的 Threads 访问连续地址：

```
Thread 0  -> a[0]
Thread 1  -> a[1]
...
Thread 31 -> a[31]
```

这种连续访问模式允许 GPU 将多个内存请求合并为尽可能少的 DRAM 事务，极大提高有效带宽。反之，若访问是跳跃的（Strided），则会增加事务数，降低带宽利用率。

---

## 16. 实验二：Elementwise Kernel Fusion

### 数学任务
$$
y_i = \text{ReLU}(\alpha \cdot x_i + \beta)
$$

其中 ReLU 定义为 $\max(0, z)$。

### Unfused 版本（两个 Kernel）

#### Affine
```cpp
__global__ void affine_kernel(const float* x, float* tmp, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = alpha * x[i] + beta;
}
```

#### ReLU
```cpp
__global__ void relu_kernel(const float* tmp, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tmp[i] > 0.0f ? tmp[i] : 0.0f;
}
```

**数据流**：`x (HBM) → affine → tmp (HBM) → relu → y (HBM)`

中间 `tmp` 需写 HBM 并重新读取，产生额外 HBM 流量。

### Fused 版本（单 Kernel）
```cpp
__global__ void fused_affine_relu(const float* x, float* y, float alpha, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = alpha * x[i] + beta;   // 仿射输出
        y[i] = value > 0.0f ? value : 0.0f;  // 直接 ReLU
    }
}
```

中间变量 `value` 驻留在 **寄存器** 中，不读写 HBM。

**数据流**：`x (HBM) → fused (寄存器计算) → y (HBM)`

### 收益分析

| 版本 | Kernel 数 | HBM 读 | HBM 写 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| Unfused | 2 | 读 x + 读 tmp | 写 tmp + 写 y | tmp 产生额外读写 |
| Fused | 1 | 读 x | 写 y | 消除 tmp 全部访问 |

Fusion 减少了 HBM 流量（约减半），并减少了 Kernel Launch 开销。这是 **FlashAttention** 等更复杂融合技术在微观层面的体现。

> **注意**：Fusion 并非总是更好，复杂 Kernel 可能增加 Register 压力，导致 Occupancy 下降或 Spill。需要根据具体情况进行权衡。

---

## 17. 实验三：Block Size Sweep

### 目标
测量 VectorAdd 在不同 Block Size（32, 64, 128, 256, 512, 1024）下的性能，观察 **Warp 数量**、**Occupancy** 和 **调度效率** 对性能的影响。

### Block Size 与 Warp 关系

| Block Size | Warps / Block |
| :---: | :---: |
| 32 | 1 |
| 64 | 2 |
| 128 | 4 |
| 256 | 8 |
| 512 | 16 |
| 1024 | 32 |

### 为什么 Block Size 影响性能？

- **延迟隐藏**：Block 内 Warp 越多，SM 调度器更容易在等待内存访问时切换到同 Block 的其他 Warp，减少空泡。
- **资源占用**：大 Block 消耗更多寄存器/共享内存，可能限制 SM 上同时驻留的 Block 数量，降低总体 Occupancy。
- **调度开销**：Block 数量过多或过少都会影响硬件利用率。

**最优值取决于 Kernel 特点和 GPU 架构，必须通过实验确定。**

### Benchmark 框架

```cpp
float benchmark(const float* a, const float* b, float* c, int n, int block_size, int warmups, int repeats) {
    int grid_size = (n + block_size - 1) / block_size;

    // Warmup
    for (int i = 0; i < warmups; ++i)
        vector_add<<<grid_size, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    // CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i)
        vector_add<<<grid_size, block_size>>>(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeats;   // 单次平均时间
}
```

### Effective Bandwidth 计算

VectorAdd 每元素传输 3 个 float（读 A，读 B，写 C），总字节数 $= 3N \times \text{sizeof(float)}$。

$$
\text{Bandwidth (GB/s)} = \frac{3N \times \text{sizeof(float)}}{\text{Kernel Time (s)}} \times 10^{-9}
$$

### 预期结果（示例）

| Block Size | Warps | Time (ms) | Bandwidth (GB/s) |
| :---: | :---: | :---: | :---: |
| 32 | 1 | 15.2 | 120 |
| 64 | 2 | 12.8 | 143 |
| 128 | 4 | 11.0 | 166 |
| 256 | 8 | 10.5 | 174 |
| 512 | 16 | 10.6 | 172 |
| 1024 | 32 | 11.3 | 161 |

**分析**：随着 Block Size 增大，带宽先升后降，最优值通常在 128~512 之间。

> **注意**：不能仅凭 Timing 断言最优 Block Size 对应最高 Occupancy，必须使用 Profiler（如 Nsight Compute）获取实际 Occupancy、寄存器使用、内存事务数等底层指标。

---

## 18. GPU 执行路径完整图解

```
CPU Host
│
│ kernel<<<grid, block>>>
↓
CUDA Runtime / Driver
│
↓
GPU Grid
│
├── Block 0
├── Block 1
├── Block 2
└── ...
     │
     ↓
Hardware Block Scheduling
     │
 ┌───┴─────────────────────────┐
 ↓                             ↓
SM 0                           SM 1
│                              │
Resident Blocks                Resident Blocks
│
↓
Each Block split into Warps
│
├── Warp 0 (32 Threads)
├── Warp 1
└── ...
│
↓
Warp Scheduler
│
├── Arithmetic Instruction → CUDA Core / FP Pipeline
├── MMA Instruction        → Tensor Core
└── Load / Store           → Memory Pipeline
                                │
                                ▼
                     Register / Shared / L1
                                │
                                ▼
                               L2
                                │
                                ▼
                           HBM / DRAM
```

### 关键结论

- **一个 Thread Block 的 Threads 在同一个 SM 上执行**。
- **一个 SM 可同时驻留多个 Blocks**，数量受资源限制。
- **Warp 是调度执行的基本单位**（32 Threads）。
- **Thread 是软件实体，不等于物理 CUDA Core**。
- **数据流经 L1/L2/HBM**，层次清晰。

---

## 19. C++ 副轨

### Pointer vs Reference

| 特性 | Pointer (`T*`) | Reference (`T&`) |
| :--- | :--- | :--- |
| 本质 | 存储地址的变量 | 对象的别名 |
| 可为空 | 是（`nullptr`） | 否（必须绑定有效对象） |
| 可重新赋值 | 是 | 否 |
| 语法 | `*` 解引用，`&` 取地址 | 直接使用，如同对象本身 |
| 在 CUDA 中 | 广泛使用（`cudaMalloc` 等） | 较少 |
| 现代 C++ 推荐 | 用于可空/需要改变指向的场景 | 用于参数传递、避免拷贝 |

**示例**：
```cpp
float* ptr = nullptr;
cudaMalloc(&ptr, bytes);   // 传指针的地址

void fill(std::vector<float>& data) {
    for (auto& x : data) x = 1.0f;
}
```

### RAII（Resource Acquisition Is Initialization）

核心思想：将资源的生命周期与对象的生命周期绑定。

- **构造函数**：获取资源。
- **析构函数**：释放资源。

**CUDA 原始方式**：
```cpp
cudaMalloc(&ptr, size);
...
cudaFree(ptr);   // 易遗漏
```

**RAII 封装**：
```cpp
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    // 禁止拷贝（防止 Double Free）
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    // 允许移动转移所有权
    DeviceBuffer(DeviceBuffer&& other) noexcept : count_(other.count_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};
```

**使用**：
```cpp
{
    DeviceBuffer<float> d_a(N);
    // ... 使用 d_a.data()
}   // 离开作用域，析构自动 cudaFree
```

**为什么禁止拷贝**？若默认拷贝，两个对象拥有同一指针，析构时会出现两次 `cudaFree`，导致 **Double Free** 崩溃。

### STL 使用
- `std::vector`：动态连续数组，常用于 Host 数据。
- `std::ofstream`：文件输出，用于保存 Benchmark 结果。
- `range-based for`：简化遍历。

---

## 20. CPU Cache Locality 实验

### 目的
直观理解 **Memory Access Pattern** 对性能的影响，这一原则同样适用于 GPU Coalescing。

### 行主序（Row-major）访问
```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
        sum += a[i * N + j];
```
地址连续，CPU Cache 友好。

### 列主序（Column-major）访问
```cpp
for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
        sum += a[i * N + j];
```
地址跳跃（步长 N），Cache 命中率低，性能差。

### 实验代码
```cpp
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    const int N = 8192;
    std::vector<float> a(static_cast<size_t>(N) * N, 1.0f);
    volatile float sink = 0.0f;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sink += a[i * N + j];
    auto end = std::chrono::steady_clock::now();
    double time_row = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::steady_clock::now();
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            sink += a[i * N + j];
    end = std::chrono::steady_clock::now();
    double time_col = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Row-major: %.2f ms\nColumn-major: %.2f ms\nSpeedup: %.2f x\n",
           time_row, time_col, time_col / time_row);
    return 0;
}
```

**观察**：Row-major 通常比 Column-major 快 5~10 倍，体现 Cache Locality 的重要性。

---

## 21. 本周关键概念速查表

| 概念 | 简要定义 |
| :--- | :--- |
| **Grid** | 一次 Kernel Launch 的全部 Block 集合。 |
| **Block** | 在同一个 SM 上执行、可共享 Shared Memory 的 Thread 组。 |
| **Warp** | 32 个 Thread，硬件调度单位。 |
| **Thread** | 软件逻辑执行实体。 |
| **SM** | Streaming Multiprocessor，包含 Warp Scheduler、Register File、Shared Memory 等。 |
| **Coalescing** | Warp 内 Thread 访问连续 Global Memory 地址，减少 Memory Transactions。 |
| **Occupancy** | Active Warps / Max Resident Warps，影响 Latency Hiding 潜力。 |
| **Register Pressure** | 每 Thread 寄存器需求过高，导致 Occupancy 下降或 Spill。 |
| **Kernel Fusion** | 合并多个 Kernel，减少中间 HBM 流量和 Launch 开销。 |
| **Arithmetic Intensity** | FLOPs / Bytes，决定 Kernel 是 Compute-Bound 还是 Memory-Bound。 |

---

## 22. 实验结果记录模板

### 环境信息
```text
GPU: NVIDIA RTX 4090 (24 GB)
CUDA Toolkit: 12.1
CUDA Driver: 535.104.05
NVCC: 12.1
OS: Ubuntu 22.04
Compiler Flags: -O3 -std=c++17
```

### VectorAdd 实验
```text
N: 1,048,576
Block Size: 256
Grid Size: 4096
Correctness: PASS
Kernel Time: 0.085 ms
Effective Bandwidth: 148.2 GB/s
```

### Elementwise Fusion 对比（理论分析）
| 版本 | Kernel 数 | HBM 读 | HBM 写 | 理论加速 |
| :--- | :---: | :---: | :---: | :---: |
| Unfused | 2 | 读 x + 读 tmp | 写 tmp + 写 y | 1x |
| Fused | 1 | 读 x | 写 y | ~2x |

### Block Size Sweep 结果（示例）
| Block Size | Warps/Block | Time (ms) | Bandwidth (GB/s) |
| :---: | :---: | :---: | :---: |
| 32 | 1 | 15.2 | 120.5 |
| 64 | 2 | 12.8 | 143.1 |
| 128 | 4 | 11.0 | 166.4 |
| 256 | 8 | 10.5 | 174.3 |
| 512 | 16 | 10.6 | 172.7 |
| 1024 | 32 | 11.3 | 161.9 |

---

## 23. 高频面试自测题

<details>
<summary><b>Q1：Grid / Block / Warp / Thread 的关系？</b></summary>
Grid 包含多个 Block，Block 包含多个 Thread。在 NVIDIA GPU 中，Block 内 Thread 按 32 个一组组成 Warp。Block 被调度到 SM 上执行，一个 SM 可同时驻留多个 Block 和 Warp。
</details>

<details>
<summary><b>Q2：一个 Block 可以跨多个 SM 吗？</b></summary>
不可以。一个 Block 的所有 Threads 在同一个 SM 上执行，因为它们需要共享 Shared Memory 和进行 Block 内同步。
</details>

<details>
<summary><b>Q3：Thread 等于 CUDA Core 吗？</b></summary>
不等于。Thread 是软件逻辑执行实体，而 CUDA Core 是硬件执行单元。Warp 的指令由 Warp Scheduler 发射到不同的硬件 Pipeline（包括 CUDA Core、Tensor Core、Load/Store Unit 等）。
</details>

<details>
<summary><b>Q4：什么是 Coalesced Access？</b></summary>
同一 Warp 中 Threads 对 Global Memory 的访问地址连续或具有良好局部性，使 GPU 能用较少的 Memory Transactions 完成访问，提高有效带宽。
</details>

<details>
<summary><b>Q5：为什么 VectorAdd 是 Memory-Bound？</b></summary>
每个 FP32 元素需移动 12 Bytes 数据，但只执行 1 次加法（1 FLOP），Arithmetic Intensity 约 0.083 FLOP/Byte，远低于 GPU 峰值，因此受带宽限制。
</details>

<details>
<summary><b>Q6：Occupancy 高一定快吗？</b></summary>
不一定。Occupancy 只反映 Latency Hiding 的潜力，实际性能还受 Coalescing、Bank Conflict、Register Spill、ILP、Compute Pipeline 利用率等多因素影响。
</details>

<details>
<summary><b>Q7：为什么 Block Size 要 Benchmark？</b></summary>
Block Size 影响 Warps/Block、Resident Blocks、资源分配和调度效率，最优值依赖 Kernel 和 GPU 架构，必须通过实验确定。
</details>

<details>
<summary><b>Q8：为什么不能仅凭 Timing 推断 Occupancy？</b></summary>
Timing 是综合结果，未直接测量 Occupancy、寄存器使用、事务数等底层指标，需要 Profiler 工具（如 Nsight Compute）辅助分析。
</details>

<details>
<summary><b>Q9：Kernel Fusion 为什么可能加速？</b></summary>
减少 Kernel Launch 次数，更重要的是减少中间 Tensor 的 HBM 读写，提高 On-chip Memory 复用。
</details>

<details>
<summary><b>Q10：Fusion 为什么不能无限做？</b></summary>
融合会增加 Kernel 复杂度，可能引起 Register Pressure、Shared Memory 压力、Occupancy 下降和 Spill，带来性能回退。
</details>

<details>
<summary><b>Q11：CPU Cache 与 GPU Coalescing 的联系？</b></summary>
两者都强调 **Memory Access Pattern** 的重要性。CPU 依赖 Cache Line 和 Spatial Locality，GPU 依赖 Warp 级 Coalescing，本质都是通过连续访问提高 Memory System 效率。
</details>

---

## 24. 本周验收 Checklist

### CUDA 环境与基础
- [ ] `nvcc` 可用，GPU 可识别
- [ ] 能编译并运行 `hello_cuda.cu`
- [ ] 理解 `__global__`、`<<<>>>`、`blockIdx`、`threadIdx`

### VectorAdd
- [ ] 正确编写 Kernel
- [ ] 使用 `cudaMalloc`、`cudaMemcpy`、`cudaFree`
- [ ] 理解 `grid_size` 和 `block_size` 计算
- [ ] 理解 `if (i < n)` 边界检查
- [ ] 验证正确性（Golden Reference）
- [ ] 能解释为何 Memory-Bound 和 Coalesced

### Elementwise Fusion
- [ ] 编写 Affine 和 ReLU Kernel
- [ ] 编写 Fused Kernel
- [ ] 验证正确性
- [ ] 能画出 Unfused / Fused 数据流图
- [ ] 理解 Register 复用

### Benchmark
- [ ] 实现 Warmup + Repeat
- [ ] 使用 CUDA Event 计时
- [ ] 仅测量 Kernel 时间（不含 H2D/D2H）
- [ ] 遍历 Block Size 32~1024
- [ ] 导出 CSV 结果
- [ ] 计算 Effective Bandwidth
- [ ] 能解释结果趋势（不盲目归因）

### 执行模型
- [ ] 能绘制 Launch → Grid → Block → SM → Warp → Execution Unit → Memory 路径图
- [ ] 理解 Warp 调度与 Latency Hiding

### C++ 工程
- [ ] 理解 Pointer 与 Reference 区别
- [ ] 理解 RAII 与资源所有权
- [ ] 实现简单的 `DeviceBuffer` 封装
- [ ] 理解禁止拷贝的原因
- [ ] 熟练使用 `std::vector`、`std::ofstream`、range-based for

### CPU Cache 实验
- [ ] 编写并运行 Row-major vs Column-major 对比
- [ ] 观察显著加速比
- [ ] 能联系到 GPU Coalescing

---

## 25. 简历与作品集素材

### 项目描述
> **CUDA 性能工程入门实践**：在 NVIDIA GPU 上完成 VectorAdd、Elementwise Kernel Fusion、Block Size Sweep 与 Effective Bandwidth 分析。深入理解 Grid/Block/Warp/SM 执行模型、Coalesced Memory Access、Occupancy 与 Kernel Fusion 原理。实现 RAII 风格的 Device 内存管理，并使用 CUDA Event 进行精确性能测量。

### 关键技能
- CUDA C++：Kernel 编写、内存管理、性能分析
- GPU 架构：SM、Warp、Memory Hierarchy
- 性能工程：Benchmark 设计、Effective Bandwidth、微基准测试
- 现代 C++：RAII、模板、STL 容器

---

## 26. Week 03 → Week 04 衔接

本周已经建立了 **CUDA 执行模型** 和 **Memory Access** 的坚实认知。下一周将进入更高级的优化技术：

- **Shared Memory** 显式管理
- **Tiling** 分块策略
- **Bank Conflict** 分析
- **Reduction** 归约算法
- **Tiled MatMul** 实战

现在你已经能回答：

> 为什么优化算法往往要把数据从 HBM 搬到 Shared Memory，再让多个 Threads 重复利用？

因为 Shared Memory 的延迟远低于 HBM，且通过 Tiling 可以显著减少 HBM 流量——这正是 FlashAttention 等 IO-aware 算法的核心。

---

> **最终目标**：将 **Execution Model + Memory Model + Data Movement Thinking** 融会贯通，为后续学习 Triton、FlashAttention 和 LLM Kernel 优化奠定坚实根基。