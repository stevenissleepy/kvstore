#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class HNSW {
private:
    struct Node {
        uint64_t key;
        std::vector<float> vec;
        std::vector<std::vector<uint32_t>> neighbors;
        uint32_t max_layer;

        Node(uint64_t key, std::vector<float> vec, int ml) : key(key), vec(vec), max_layer(ml) {
            neighbors.resize(max_layer + 1);
        }
    };

public:
    HNSW(int M = 24, int M_max = 38, int ef = 30, int m_l = 6);
    void insert(uint64_t key, const std::vector<float> &vec);
    void erase(uint64_t key, const std::vector<float> &vec);
    std::vector<uint64_t> query(const std::vector<float> &q, int k);

    void putFile(const std::string &root = "./data/hnsw_data");
    void loadFile(const std::string &root = "./data/hnsw_data");

private:
    int search_layer_greedy(const std::vector<float> &q, int layer, int ep);
    std::vector<std::pair<float, int>> search_layer(const std::vector<float> &q, int layer, int ep = -1);

    int random_layer();
    void connect(int id, int neighbor_id, int layer);
    float distance(const std::vector<float> &a, const std::vector<float> &b);
    float similarity_cos(const std::vector<float> &a, const std::vector<float> &b);
    bool is_deleted(uint64_t key, const std::vector<float> &vec);

    void put_file_header(const std::string &root);
    void put_file_deleted_nodes(const std::string &root);
    void put_file_nodes(const std::string &root);

    void load_file_header(const std::string &root, uint32_t &size, uint32_t &dim);
    void load_file_deleted_nodes(const std::string &root, const uint32_t &dim);
    void load_file_nodes(const std::string &root, const uint32_t &size, const uint32_t &dim);

private:
    std::vector<Node> nodes;
    std::vector<std::pair<uint64_t, std::vector<float>>> deleted_nodes;

    uint32_t M;
    uint32_t M_max;
    uint32_t ef_construction;
    uint32_t m_L;
    uint32_t top_layer;
    uint32_t entry_point;
};
