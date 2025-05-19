#include "kvstore.h"
#include <chrono>
#include <iostream>
#include <functional>

void measure_operation(const std::string &operation, int num_operations, const std::function<void()> &func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); 
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << operation << ": " << num_operations << " operations, "
              << "总耗时: "   << duration << " us, "
              << "平均耗时: " << (duration / num_operations) << " us/op, "
              << "吞吐量: "   << (num_operations * 1e6 / duration) << " ops/s" << std::endl;
}

void performance_test(const std::string &dir, int num_operations) {
    KVStore store(dir);
    std::string value = std::string(100, 'x'); 

    /* PUT 操作 */
    measure_operation("PUT", num_operations, [&]() {
        for (int i = 0; i < num_operations; ++i) {
            store.put(i + 1, value);
        }
    });

    /* GET 操作 */ 
    measure_operation("GET", num_operations, [&]() {
        for (int i = 0; i < num_operations; ++i) {
            store.get(i + 1);
        }
    });

    /* DEL 操作 */
    measure_operation("DEL", num_operations, [&]() {
        for (int i = 0; i < num_operations; ++i) {
            store.del(i + 1);
        }
    });
}

int main() {
    std::string dir = "./data";
    int num_operations = 100000; // 测试 100,000 次操作

    performance_test(dir, num_operations);

    return 0;
}
