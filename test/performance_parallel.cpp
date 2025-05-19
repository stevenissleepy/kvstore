#include "kvstore.h"
#include <chrono>
#include <iostream>
#include <functional>
#include <fstream>

void measure_put(KVStore& store, int num_operations, const std::vector<std::string>& texts, const std::vector<std::vector<float>>& vectors);
void measure_get(KVStore& store, int num_operations);
void measure_del(KVStore& store, int num_operations);
void performance_test(int num_operations);
std::vector<std::string> get_texts(int num, std::string& file);
std::vector<std::vector<float>> get_vectors(int num, std::string& file); 

int main() {
    int num_operations = 50000;
    performance_test(num_operations);

    return 0;
}

void performance_test(int num_operations) {
    std::string dir = "./data";
    std::string txt_file = "./data/cleaned_text_100k.txt";
    std::string vec_file = "./data/embedding_100k.txt";
    
    KVStore store(dir);
    store.reset();
    std::vector<std::string> texts = get_texts(num_operations, txt_file);
    std::vector<std::vector<float>> vectors = get_vectors(num_operations, vec_file);

    measure_put(store, num_operations, texts, vectors);
    measure_get(store, num_operations);
    measure_del(store, num_operations);
}

void measure_put(KVStore& store, int num_operations, const std::vector<std::string>& texts, const std::vector<std::vector<float>>& vectors) {
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

std::vector<std::string> get_texts(int num, std::string& file) {
    std::vector<std::string> texts;
    std::ifstream infile(file);
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue; // 跳过空行
        texts.push_back(line);
        if (texts.size() >= num) break; // 读取指定数量的文本
    }
    return texts;
}

std::vector<std::vector<float>> get_vectors(int num, std::string& file) {
    std::vector<std::vector<float>> vectors;
    std::ifstream infile(file);
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (line.front() == '[') line = line.substr(1);
        if (!line.empty() && line.back() == ']') line.pop_back();

        std::vector<float> vec;
        size_t start = 0, end;
        while ((end = line.find(',', start)) != std::string::npos) {
            vec.push_back(std::stof(line.substr(start, end - start)));
            start = end + 1;
        }
        // 最后一个数字
        if (start < line.size())
            vec.push_back(std::stof(line.substr(start)));

        vectors.push_back(std::move(vec));
        if (vectors.size() >= num) break;
    }
    return vectors;
}
