#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class HNSW {
private:
    struct Node {
        int id;
        std::vector<float> vec;
        std::unordered_map<int, std::vector<int>> neighbors; // 各层的邻居
    };

    int max_layers;                      // 最大层数
    int M;                               // 每个节点最大连接数
    int ef_construction;                 // 动态候选集大小
    std::unordered_map<int, Node> nodes; // 所有节点
    int enter_point;                     // 入口点
    std::mt19937 rng;                    // 随机数生成器

public:
    HNSW(int ml = 5, int m = 16, int ef = 200);

    void insert(int id, const std::vector<float> &vec);
    std::vector<int> query(const std::vector<float> &q, int k);

private:
    int random_level();

    // 在指定层搜索
    std::vector<int> search_layer(const std::vector<float> &q, int ep, int ef, int layer);

    // 连接两个节点
    void connect(int a, int b, int layer);

    // 计算欧氏距离
    float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b);
};
