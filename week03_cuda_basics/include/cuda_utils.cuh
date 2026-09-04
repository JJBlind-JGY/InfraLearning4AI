// GPU Execution Model    

    //                 CPU HOST
    //                    │
    //                    │
    //           kernel<<<G, B>>>()
    //                    │
    //                    ▼
    //          CUDA Runtime / Driver
    //                    │
    //                    ▼
    //                 GPU GRID
    //                    │
    //     ┌──────────────┼──────────────┐
    //     ▼              ▼              ▼
    //  Block 0        Block 1         Block 2
    //     │              │              │
    //     └──── Hardware Block Scheduling ──┐
    //                                       │
    //               ┌───────────────────────┴────────────────────┐
    //               ▼                                            ▼
    //             SM 0                                          SM 1
    //     ┌─────────────────┐                          ┌─────────────────┐
    //     │ Resident Blocks │                          │ Resident Blocks │
    //     └────────┬────────┘                          └────────┬────────┘
    //              ▼                                            ▼
    //            Warps                                        Warps
    //     Warp 0 / Warp 1 ...                         Warp 0 / Warp 1 ...
    //              │
    //              ▼
    //       Warp Scheduler
    //              │
    //     ┌────────┼─────────┐
    //     ▼        ▼         ▼
    // CUDA Core Tensor Core Load/Store
    //                        │
    //                        ▼
    //                  Registers
    //                  Shared / L1
    //                        │
    //                        ▼
    //                       L2
    //                        │
    //                        ▼
    //                   HBM / DRAM



#pragma once        // 这是一个预处理指令（以 # 开头）。它的作用是保证这个头文件在编译过程中只被包含一次
#include<cstdio>
#include<cstdlib>
#include<cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error : %s : (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); std::exit(EXIT_FAILURE);}} while (0)

// RAII 自动管理显存
template<typename T>
class DeviceBuffer {
    public:
    // 1. 构造函数:申请显存
    // explicit：这是一个重要关键字。它禁止编译器进行“隐式转换”。比如，如果你写了 DeviceBuffer<float> d_buf = 100;，如果没有 explicit，编译器会悄悄把 100 转成 size_t 然后调用构造。explicit 强制你必须写成 DeviceBuffer<float> d_buf(100);，防止手滑写出莫名其妙的代码
    explicit DeviceBuffer(size_t count): count_(count), ptr_(nullptr) {
        if(count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    // 2. 析构函数:自动释放显存
    ~DeviceBuffer() {
        // RAII 的精髓：ptr_ 在构造时申请显存，在析构时释放。只要对象活着，显存就一定被占用；对象死了，显存一定被释放。 再也无需手动 cudaFree
        if (ptr_) {
            CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;     // 避免野指针
        }
    }

    // 3. 禁止拷贝(放置 Double Free)
    // 拷贝构造函数 和 拷贝赋值运算符。后面加 = delete，意思就是禁止拷贝
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // 4. 允许移动 (转移所有权)
    // 可以将其放到容器中, 或者通过函数返回
    // 移动构造函数（Move Constructor）。参数中的 && 代表“右值引用”，通俗讲就是“一个即将被销毁的临时对象”
    // noexcept：承诺这个函数不会抛出异常。这能让 C++ 标准库（如 std::vector）在扩容时使用高效的移动构造，而不是拷贝构造
    // 逻辑：它直接把 other 的显存指针（ptr_）偷过来（ptr_(other.ptr_)），然后把 other 的指针置空（other.ptr_ = nullptr）。这样只有新对象拥有显存，旧对象变成了空壳
    // 对应的移动赋值运算符：与拷贝赋值逻辑类似，释放自己的旧资源，偷走对方的资源
    DeviceBuffer(DeviceBuffer&& other) noexcept: count_(other.count_), ptr_(other.ptr_) {
        other.count_ = 0;
        other.ptr_ = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if(this != &other) {
            if(ptr_) CUDA_CHECK(cudaFree(ptr_));
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.count_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // 5. 获取原始指针 (仅kernel调用)
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    // 6. 获取大小
    size_t size() const { return count_; }
    
    private:
        size_t count_;
        T* ptr_;
};
