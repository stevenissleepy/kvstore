#pragma once

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class HNSW {
private:
    struct Node {
        int key;
        std::vector<float> vec;
        std::vector<int> neighbors;
    };

    int max_layers;                                    // 最大层数
    int M;                                             // 新插入节点最大连接数
    int M_max;                                         // 每个节点的最大连接数
    int ef_construction;                               // 动态候选集大小
    int q;                                             // 增长率
    std::vector<std::unordered_map<int, Node>> layers; // 各层的节点
    std::mt19937 rng;                                  // 随机数生成器

public:
    HNSW(int max_layers = 6, int M = 6, int M_max = 8, int ef_construction = 30, int q = 0.5);

    void insert(int id, const std::vector<float> &vec);
    std::vector<int> query(const std::vector<float> &q, int k);

private:
    int random_level();

    std::vector<int> search_layer(const std::vector<float> &q, int k, int ep, int layer);

    void connect(int a, int b, int layer);

    float distance(const std::vector<float> &a, const std::vector<float> &b);
    float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b);
    float similarity_cos(const std::vector<float> &a, const std::vector<float> &b);
};
