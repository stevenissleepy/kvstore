#pragma once

#include <random>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class HNSW {
private:
    struct Node {
        int id;
        std::vector<float> vec;
        std::vector<int> neighbors;
    };

    int max_layers;                                    // 最大层数
    int top_layer;                                     // 当前顶层
    int M;                                             // 每个节点最大连接数
    int ef_construction;                               // 动态候选集大小
    std::vector<std::unordered_map<int, Node>> layers; // 各层的节点
    std::mt19937 rng;                                  // 随机数生成器

public:
    HNSW(int ml = 6, int m = 12, int ef = 60);

    void insert(int id, const std::vector<float> &vec);
    std::vector<int> query(const std::vector<float> &q, int k);
    void check_layers();

private:
    int random_level();

    std::vector<int> search_layer(const std::vector<float> &q, int k, int ep, int layer);

    void connect(int a, int b, int layer);

    float distance(const std::vector<float> &a, const std::vector<float> &b);
    float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b);
    float similarity_cos(const std::vector<float> &a, const std::vector<float> &b);
};
