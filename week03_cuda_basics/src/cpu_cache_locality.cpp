#include<chrono>    // C++11 引入的时间库。比古老的 clock() 精度高得多，且支持跨平台使用
#include<cstdio>
#include<cstdlib>
#include<vector>

int main() {
    const int N = 8192; // 8192 * 8192 个元素 float, 共计 256MB
    // 分配二维数组(展平为一维, C++默认行主序)
    std::vector<float> a(static_cast<size_t>(N) * N, 1.0f);         // std::vector<float> a(数量, 初始值)：这个构造函数接受两个参数。第一个是元素个数（这里展成了一维），第二个是初始值（1.0f）。
    

    //volatile：这是一个极其关键的关键字。它告诉编译器：“这个变量随时可能被外部（比如另一个线程或硬件）改变，禁止对它做任何优化。”
    // 为什么这里必须用：因为你的循环是 sink += a[...]。如果不加 volatile，编译器在 -O3 优化下会思考：“这家伙疯狂累加，但最后又不用这个结果，那我干脆把整个循环都删掉！” 加了 volatile，编译器必须老老实实执行每一次累加，确保我们真的测到了内存访问时间
    volatile float sink = 0.0f; // volatile 防止编译器优化掉整个循环

    // test 1: Row-major (连续访问)
    auto start = std::chrono::steady_clock::now();      // auto：让编译器自动推导类型，实际返回类型是 std::chrono::time_point<std::chrono::steady_clock>
    // std::chrono::steady_clock：单调时钟。它保证时间只会往前走，不会受系统时间调整（比如手动改系统时间）影响，非常适合测性能
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sink += a[i * N + j];  // 行主序（Row-major） 公式。内存中排列是 a[0][0], a[0][1], a[0][2] ...。当 i=0 时，j 递增，地址连续，CPU 预取（Prefetcher）会把后续数据提前加载到 Cache，极快
        }
    }
    auto end = std::chrono::steady_clock::now();
    // end - start：返回 std::chrono::duration 时长对象
    // duration<double, std::milli>：将时间间隔表示为 double 类型的毫秒数
    // count()：返回时间间隔的毫秒数, 取出数值
    double time_row = std::chrono::duration<double, std::milli>(end - start).count();      
    printf("Row-major time: %.2f ms\n", time_row);

    // test 2: Column-major (跳跃访问, 大跨度)
    start = std::chrono::steady_clock::now();
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; i++) {
    // Column-major 的索引是 a[i * N + j]，但循环顺序换成了 j 在外层，i 在内层。此时地址跳跃步长是 N（8192 个 float，即 32KB），远超 CPU 的 Cache Line（64 字节），每次访问都要去主内存取，所以慢得多
            sink += a[i * N + j];
        }
    }
    end = std::chrono::steady_clock::now();
    double time_col = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Column-major time: %.2f ms\n", time_col);

    printf("Speedup (Row-major / Column-major) = %.2f\n", time_row / time_col);

    printf("Sink = %f \n", sink);
    return 0;
}