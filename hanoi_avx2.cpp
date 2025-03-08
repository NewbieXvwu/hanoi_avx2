#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <intrin.h>
#include <atomic>
#include <thread>
#include <cstdio>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#pragma warning(disable : 4996 5038 5002)

using namespace std;
using namespace std::chrono;

// 调整为 AVX2 友好的 32 字节对齐结构
alignas(32) constexpr struct Direction {
    char from[8], to[8];
} ODD_DIR  = {{'C','B','A','C','B','A','C','B'}, {'B','A','C','B','A','C','B','A'}},
  EVEN_DIR = {{'B','C','A','B','C','A','B','C'}, {'C','A','B','C','A','B','C','A'}};

// 避免伪共享的线程局部计数器
struct alignas(64) ThreadLocalCounter {
    atomic<uint64_t> count;
    char padding[64 - sizeof(atomic<uint64_t>)]; // 填充到完整缓存行

    ThreadLocalCounter() : count(0) {}
};

// 为每个线程创建专用计数器
ThreadLocalCounter* thread_counters = nullptr;

// 全局计数器用于最终统计
atomic<uint64_t> moves_completed(0);

// 使用模板参数 N 避免硬编码
template <int N>
__forceinline void process_move(uint64_t m, atomic<uint64_t>& local_counter) {
    // 提前预取下一批数据
    _mm_prefetch(reinterpret_cast<const char*>(&ODD_DIR) + 64, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&EVEN_DIR) + 64, _MM_HINT_T0);

    if (m > ((1ULL << N) - 1)) return; // 安全检查，防止越界

    unsigned long index;
    _BitScanForward64(&index, m);
    const int c = static_cast<int>(index) + 1;

    const uint64_t mask = -(uint64_t)(c <= N);
    const uint64_t k = (m >> c) + 1;

    // 使用联合体替代向量提取指令
    union AVXConverter {
        __m256i vec;
        int16_t arr[16];
        constexpr AVXConverter(__m256i v) : vec(v) {}
    };

    const AVXConverter dirs = (c & 1) ? 
        AVXConverter(_mm256_load_si256((const __m256i*)&ODD_DIR)) :
        AVXConverter(_mm256_load_si256((const __m256i*)&EVEN_DIR));

    const int rem = ((k - 1) & 0x3) & static_cast<int>(mask);

    // 确保 rem 值在有效范围内
    if (rem >= 0 && rem < 8) {
        int16_t result = dirs.arr[rem] + dirs.arr[rem + 4];
        // 防止编译器完全优化掉计算
        if (result == 'Z') {
            local_counter.fetch_add(1, memory_order_relaxed);
        }
    }
}

// 使用AVX2并行处理多个移动
template <int N>
__forceinline void process_moves_vectorized(uint64_t base_m, atomic<uint64_t>& local_counter) {
    // 生成前 4 个连续的移动索引： base_m + {0,1,2,3}
    __m256i indices_low = _mm256_add_epi64(
        _mm256_set1_epi64x(base_m),
        _mm256_set_epi64x(3, 2, 1, 0) // 从高到低依次为：3,2,1,0
    );
    // 生成后 4 个连续的移动索引： base_m + {4,5,6,7}
    __m256i indices_high = _mm256_add_epi64(
        _mm256_set1_epi64x(base_m),
        _mm256_set_epi64x(7, 6, 5, 4) // 从高到低依次为：7,6,5,4
    );

    // 显式展开提取每个元素（要求立即数索引）
    uint64_t m_low0 = _mm256_extract_epi64(indices_low, 0);
    uint64_t m_low1 = _mm256_extract_epi64(indices_low, 1);
    uint64_t m_low2 = _mm256_extract_epi64(indices_low, 2);
    uint64_t m_low3 = _mm256_extract_epi64(indices_low, 3);

    uint64_t m_high0 = _mm256_extract_epi64(indices_high, 0);
    uint64_t m_high1 = _mm256_extract_epi64(indices_high, 1);
    uint64_t m_high2 = _mm256_extract_epi64(indices_high, 2);
    uint64_t m_high3 = _mm256_extract_epi64(indices_high, 3);

    _mm_prefetch(reinterpret_cast<const char*>(&m_low0) + 64, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&m_high0) + 64, _MM_HINT_T0);
    process_move<N>(m_low0, local_counter);
    process_move<N>(m_high0, local_counter);

    _mm_prefetch(reinterpret_cast<const char*>(&m_low1) + 64, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&m_high1) + 64, _MM_HINT_T0);
    process_move<N>(m_low1, local_counter);
    process_move<N>(m_high1, local_counter);

    _mm_prefetch(reinterpret_cast<const char*>(&m_low2) + 64, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&m_high2) + 64, _MM_HINT_T0);
    process_move<N>(m_low2, local_counter);
    process_move<N>(m_high2, local_counter);

    _mm_prefetch(reinterpret_cast<const char*>(&m_low3) + 64, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&m_high3) + 64, _MM_HINT_T0);
    process_move<N>(m_low3, local_counter);
    process_move<N>(m_high3, local_counter);
}

int main() {
    int n;
    cout << "请输入汉诺塔层数: ";
    cin >> n;

    if (n > 46) {
        cerr << "警告：层数超过安全限制，可能产生溢出！" << endl;
        return 1;
    }

    const uint64_t total = (1ULL << n) - 1;
    // 每个 chunk 处理 2^16 个移动
    const uint64_t blockSize = 1ULL << 16;
    // 总 chunk 数 = (total+1 + blockSize - 1) / blockSize，注意 total+1 即为 2^n
    uint64_t totalChunks = (((total + 1) + blockSize - 1) / blockSize);

    atomic<uint64_t> chunks_done{0};       // 全局已完成 chunk 数
    atomic<bool> processing_done{false};   // 处理是否完成标志

    // 在 Windows 下启用虚拟终端处理以支持 ANSI 转义序列
    #ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
    #endif

    // 启动进度条线程
    std::thread progress_thread([&]() {
        auto prog_start = steady_clock::now();
        while (!processing_done.load(memory_order_relaxed)) {
            double elapsed = duration<double>(steady_clock::now() - prog_start).count();
            uint64_t doneChunks = chunks_done.load(memory_order_relaxed);
            double fraction = static_cast<double>(doneChunks) / totalChunks;
            if (fraction > 1.0) fraction = 1.0;
            double percent = fraction * 100.0;
            double est_total = (fraction > 0.0 ? elapsed / fraction : 0.0);
            double remaining = (fraction > 0.0 ? est_total - elapsed : 0.0);
            int barWidth = 20;
            int pos = static_cast<int>(barWidth * fraction);
            std::string bar = "[";
            for (int i = 0; i < barWidth; i++) {
                if (i < pos)
                    bar += "=";
                else if (i == pos)
                    bar += ">";
                else
                    bar += " ";
            }
            bar += "]";
            // 输出彩色且动态刷新的进度条（使用 ANSI 转义序列）
            printf("\r\033[32m%s\033[0m \033[33m%.2f%%\033[0m | 剩余时间: \033[36m%.3fs\033[0m | 耗时: \033[35m%.3fs\033[0m",
                   bar.c_str(), percent, remaining, elapsed);
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        double final_elapsed = duration<double>(steady_clock::now() - prog_start).count();
        // 最终进度条显示 100%
        printf("\r\033[32m[====================]\033[0m \033[33m100.00%%\033[0m | 剩余时间: \033[36m0.000s\033[0m | 耗时: \033[35m%.3fs\033[0m\n", final_elapsed);
        fflush(stdout);
    });

    cout << "检测到 " << omp_get_num_procs() << " 个处理器核心" << endl;
    omp_set_num_threads(omp_get_num_procs());

    // 初始化线程局部计数器
    thread_counters = new ThreadLocalCounter[omp_get_num_procs()];

    auto start_time = high_resolution_clock::now();

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        auto& local_counter = thread_counters[thread_id].count;

        if (thread_id == 0) {
            cout << "实际启动线程数: " << num_threads << endl;
        }

        // 改进的负载均衡策略（guided 调度）
        #pragma omp for schedule(guided) nowait
        for (int64_t chunk = 0; chunk <= static_cast<int64_t>(total >> 16); ++chunk) {
            const uint64_t start_idx = static_cast<uint64_t>(chunk) << 16;
            const uint64_t end_idx = min(start_idx + blockSize - 1, total);

            for (uint64_t m = start_idx; m <= end_idx; m += 16) {
                process_moves_vectorized<46>(m, local_counter);
                process_moves_vectorized<46>(m + 8, local_counter);
            }
            // 更新已完成的 chunk 数
            chunks_done.fetch_add(1, memory_order_relaxed);
        }

        #pragma omp critical
        {
            moves_completed += local_counter.load(memory_order_relaxed);
        }
    }

    auto end_time = high_resolution_clock::now();
    processing_done.store(true, memory_order_relaxed);
    progress_thread.join();

    // 释放线程局部计数器内存
    delete[] thread_counters;

    duration<double, milli> elapsed = end_time - start_time;
    cout << "计算完成，耗时: " << elapsed.count() << " 毫秒" << endl;

    auto speed = static_cast<double>(total) / (elapsed.count() / 1000.0) / 1000000.0;
    cout << "处理速度: " << speed << " 百万步/秒" << endl;

    printf("\n按任意键退出...");
    getchar();
    getchar();
    return 0;
}
