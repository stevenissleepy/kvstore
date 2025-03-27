#include "kvstore.h"
#include <chrono>
#include <iostream>

void performance_test(const std::string &dir, int num_operations) {
    KVStore store(dir);

    std::string value = std::string(100, 'x');

    // 测试 PUT 操作
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.put(i+1, value);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto put_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "PUT: " << num_operations << " operations, "
              << "总耗时: " << put_duration << " us, "
              << "吞吐量: " << (num_operations * 1e6 / put_duration) << " ops/sec, "
              << "平均时延: " << (put_duration / num_operations) << " us/op" << std::endl;

    // 测试 GET 操作
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.get(i+1);
    }
    end = std::chrono::high_resolution_clock::now();
    auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GET: " << num_operations << " operations, "
              << "总耗时: " << get_duration << " us, "
              << "吞吐量: " << (num_operations * 1e6 / get_duration) << " ops/sec, "
              << "平均时延: " << (get_duration / num_operations) << " us/op" << std::endl;

    // 测试 DEL 操作
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.del(i+1);
    }
    end = std::chrono::high_resolution_clock::now();
    auto del_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "DEL: " << num_operations << " operations, "
              << "总耗时: " << del_duration << " us, "
              << "吞吐量: " << (num_operations * 1e6 / del_duration) << " ops/sec, "
              << "平均时延: " << (del_duration / num_operations) << " us/op" << std::endl;
}

int main() {
    std::string dir = "./data";
    int num_operations = 1000000; // 测试 100,000 次操作

    performance_test(dir, num_operations);

    return 0;
}