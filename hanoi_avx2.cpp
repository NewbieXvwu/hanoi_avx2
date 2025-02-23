#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <intrin.h>

#ifdef _WIN32
#include <windows.h>
#endif

#pragma warning(disable : 4996 5038 5002)

using namespace std;
using namespace std::chrono;

// 调整为AVX2友好的32字节对齐结构
alignas(32) constexpr struct Direction {
    char from[8], to[8];
} ODD_DIR  = {{'C','B','A','C','B','A','C','B'}, {'B','A','C','B','A','C','B','A'}}, 
  EVEN_DIR = {{'B','C','A','B','C','A','B','C'}, {'C','A','B','C','A','B','C','A'}};

template <int N>
__forceinline void process_move(uint64_t m) {
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
    
    // 直接访问数组元素
    volatile int16_t _ = dirs.arr[rem] + dirs.arr[rem + 4];
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
    
    // 设置线程数为物理核心数
    const int physical_cores = omp_get_num_procs();
    omp_set_num_threads(physical_cores);

    auto start = high_resolution_clock::now();

    #pragma omp parallel
    {
        #ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), 1 << omp_get_thread_num());
        #endif

        #pragma omp for schedule(static, 1<<16) nowait
        for (int64_t chunk = 0; chunk <= static_cast<int64_t>(total >> 18); ++chunk) 
        {
            for (int batch = 0; batch < 4; ++batch) {
                const uint64_t start = (static_cast<uint64_t>(chunk) << 18) + (batch << 14);
                const uint64_t end = min(start + (1ULL << 14) - 1, total);
                
                // 展开为独立调用
                for (uint64_t m = start; m <= end; m += 16) {
                    process_move<46>(m);   process_move<46>(m+1);
                    process_move<46>(m+2); process_move<46>(m+3);
                    process_move<46>(m+4); process_move<46>(m+5);
                    process_move<46>(m+6); process_move<46>(m+7);
                    process_move<46>(m+8); process_move<46>(m+9);
                    process_move<46>(m+10);process_move<46>(m+11);
                    process_move<46>(m+12);process_move<46>(m+13);
                    process_move<46>(m+14);process_move<46>(m+15);
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    
    // 计算并输出耗时
    duration<double, milli> elapsed = end - start;
    cout << "计算完成，耗时: " 
         << elapsed.count() << " 毫秒" 
         << endl;
    printf("\nPress any key to exit...");
    getchar();
    getchar();
    return 0;
}