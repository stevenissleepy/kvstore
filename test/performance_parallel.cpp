#include "kvstore.h"
#include <chrono>
#include <iostream>
#include <functional>
#include <fstream>

const int max_num = 49998;

void measure_put(KVStore& store, int num_operations);
void measure_get(KVStore& store, int num_operations);
void measure_del(KVStore& store, int num_operations);
void performance_test(int num_operations);

std::vector<std::string> texts;
std::vector<std::vector<float>> vectors;
void get_texts(int num, std::string file);
void get_vectors(int num, std::string file); 

int main() {
    int num_operations = 49998;
    get_texts(num_operations, "./data/cleaned_text_100k.txt");
    get_vectors(num_operations, "./data/embedding_100k.txt");
    performance_test(num_operations);

    return 0;
}

void performance_test(int num_operations) {
    KVStore store("./data");
    store.reset();

    measure_put(store, num_operations);
    measure_get(store, num_operations);
    measure_del(store, num_operations);
}

void measure_put(KVStore& store, int num_operations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.put(i, texts[i]);
        store.put(i, vectors[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "PUT" << ": " << num_operations << " operations, "
              << "总耗时: "   << duration << " us, "
              << "平均耗时: " << (duration / num_operations) << " us/op, "
              << "吞吐量: "   << (num_operations * 1e6 / duration) << " ops/s" << std::endl;
}

void measure_get(KVStore& store, int num_operations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.get(i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GET" << ": " << num_operations << " operations, "
              << "总耗时: "   << duration << " us, "
              << "平均耗时: " << (duration / num_operations) << " us/op, "
              << "吞吐量: "   << (num_operations * 1e6 / duration) << " ops/s" << std::endl;
}

void measure_del(KVStore& store, int num_operations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        store.del(i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "DEL" << ": " << num_operations << " operations, "
              << "总耗时: "   << duration << " us, "
              << "平均耗时: " << (duration / num_operations) << " us/op, "
              << "吞吐量: "   << (num_operations * 1e6 / duration) << " ops/s" << std::endl;
}

void get_texts(int num, std::string file) {
    num = std::min(num, max_num);

    std::ifstream infile(file);
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue; // 跳过空行
        texts.push_back(line);
        if (texts.size() >= num) break; // 读取指定数量的文本
    }
}

void get_vectors(int num, std::string file) {
    num = std::min(num, max_num);

    std::ifstream infile(file);
    std::string line;
    int line_no = 0;
    while (std::getline(infile, line)) {
        ++line_no;
        if (line_no % 2 == 0) continue; // 只处理奇数行
        if (line.empty()) continue;
        if (line.front() == '[') line = line.substr(1);
        if (!line.empty() && line.back() == ']') line.pop_back();

        std::vector<float> vec;
        size_t start = 0, end;
        while ((end = line.find(',', start)) != std::string::npos) {
            vec.push_back(std::stof(line.substr(start, end - start)));
            start = end + 1;
        }
        if (start < line.size())
            vec.push_back(std::stof(line.substr(start)));

        vectors.push_back(std::move(vec));
        if (vectors.size() >= num) break;
    }
}
