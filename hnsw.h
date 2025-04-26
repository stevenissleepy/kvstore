#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

class HNSW {
private:
    struct Node {
        uint64_t key;
        std::vector<float> vec;
        std::vector<std::vector<int>> neighbors;
        int max_layer;

        Node(uint64_t key, std::vector<float> vec, int ml) : key(key), vec(vec), max_layer(ml) {
            neighbors.resize(max_layer + 1);
        }
    };

public:
    HNSW(int M = 6, int M_max = 8, int ef = 30, int m_l = 6);
    void insert(uint64_t key, const std::vector<float> &vec);
    std::vector<uint64_t> query(const std::vector<float> &q, int k);

private:
    int search_layer_greedy(const std::vector<float> &q, int layer, int ep);
    std::vector<std::pair<float, int>> search_layer(const std::vector<float> &q, int layer, int ep = -1);

    int random_layer();
    void connect(int id, int neighbor_id, int layer);
    float distance(const std::vector<float> &a, const std::vector<float> &b);
    float similarity_cos(const std::vector<float> &a, const std::vector<float> &b);

private:
    std::vector<Node> nodes;
    int entry_point;
    int top_layer;
    const int M;
    const int M_max;
    const int ef_construction;
    const int m_L;
};
