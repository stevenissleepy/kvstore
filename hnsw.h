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
        int max_level;
    };

private:
    std::vector<Node> nodes;
    int entry_point = -1;
    int max_level   = 0;
    const int M;
    const int Mmax;
    const int ef_construction;
    const float mL;

    float distance(const std::vector<float> &a, const std::vector<float> &b);
    int search_layer_greedy(const std::vector<float> &q, int layer, int entry_point_id);
    std::vector<std::pair<float, int>> search_layer(const std::vector<float> &q, int layer, int entry_point_id = -1);

public:
    HNSW(int m = 4, int mmax = 6, int ef = 40, float mlParam = 1.0f / log(6.0f));
    void insert(uint64_t key, const std::vector<float> &vec);
    std::vector<std::pair<uint64_t, float>> query(const std::vector<float> &q, int k);
};
