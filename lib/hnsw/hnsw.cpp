#include "hnsw.h"

#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>

HNSW::HNSW(int m, int M_max, int ef, int m_L) :
    M(m),
    M_max(M_max),
    ef_construction(ef),
    m_L(m_L),
    top_layer(0),
    entry_point(-1) {}

inline int HNSW::random_layer() {
    return -log((rand() / (float)RAND_MAX)) * m_L;
}

inline void HNSW::connect(int id, int neighbor_id, int layer) {
    auto &current_node  = nodes[id];
    auto &neighbor_node = nodes[neighbor_id];

    /* 连接两个 node */
    current_node.neighbors[layer].push_back(neighbor_id);
    neighbor_node.neighbors[layer].push_back(id);

    /* 如果邻居节点的 neighbor 数量大于 M_max */
    if (neighbor_node.neighbors[layer].size() > M_max) {
        /* 找到最远的邻居 */
        auto &neighbors     = neighbor_node.neighbors[layer];
        auto max_it         = std::max_element(neighbors.begin(), neighbors.end(), [&](int a, int b) {
            return distance(nodes[neighbor_id].vec, nodes[a].vec) < distance(nodes[neighbor_id].vec, nodes[b].vec);
        });
        int max_neighbor_id = *max_it;

        /* 删除最远的邻居 */
        neighbors.erase(max_it);
        auto &max_neighbor = nodes[max_neighbor_id];
        auto it = std::find(max_neighbor.neighbors[layer].begin(), max_neighbor.neighbors[layer].end(), neighbor_id);
        if (it != max_neighbor.neighbors[layer].end()) {
            max_neighbor.neighbors[layer].erase(it);
        }
    }
}

/* 余弦相似度越大，距离越短 */
inline float HNSW::distance(const std::vector<float> &a, const std::vector<float> &b) {
    return -similarity_cos(a, b);
}

float HNSW::similarity_cos(const std::vector<float> &a, const std::vector<float> &b) {
    int n = a.size();

    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
        sum1 += a[i] * a[i];
        sum2 += b[i] * b[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

bool HNSW::is_deleted(uint64_t key, const std::vector<float> &vec) {
    for (const auto &deleted_node : deleted_nodes) {
        if (deleted_node.first == key && deleted_node.second == vec) {
            return true;
        }
    }
    return false;
}

int HNSW::search_layer_greedy(const std::vector<float> &q, int layer, int ep) {
    int current_id     = ep;
    Node &current_node = nodes[current_id];
    int current_dist   = distance(q, current_node.vec);

    while (true) {
        bool found_closer = false;

        /* 找出邻居中最近的节点 */
        for (int neighbor_id : current_node.neighbors[layer]) {
            if (is_deleted(nodes[neighbor_id].key, nodes[neighbor_id].vec))
                continue;

            float neighbor_dist = distance(q, nodes[neighbor_id].vec);

            if (neighbor_dist < current_dist) {
                current_id   = neighbor_id;
                current_dist = neighbor_dist;
            }
        }

        /* 没有更近节点时终止循环 */
        if (!found_closer)
            break;
    }

    return current_id;
}

std::vector<std::pair<float, int>> HNSW::search_layer(const std::vector<float> &q, int layer, int ep) {
    using Candidate = std::pair<float, int>;
    auto cmp        = [](const Candidate &a, const Candidate &b) { return a.first > b.first; };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidates(cmp);
    std::priority_queue<std::pair<float, int>> topResults;
    std::unordered_set<int> visited;
    ep = ep == -1 ? ep : entry_point;

    float d = distance(q, nodes[ep].vec);
    candidates.emplace(d, ep);
    topResults.emplace(d, ep);
    visited.insert(ep);

    while (!candidates.empty()) {
        auto [current_dist, current_id] = candidates.top();
        candidates.pop();

        /* 如果当前节点是删除的节点，跳过 */
        if (is_deleted(nodes[current_id].key, nodes[current_id].vec))
            continue;

        /* 剪枝 */
        if (topResults.size() >= ef_construction && current_dist > topResults.top().first)
            continue;

        /* 遍历当前节点的邻居 */
        for (int neighbor_id : nodes[current_id].neighbors[layer]) {

            if (visited.find(neighbor_id) != visited.end())
                continue;
            visited.insert(neighbor_id);

            float neighbot_dist = distance(q, nodes[neighbor_id].vec);
            if (topResults.size() < ef_construction || neighbot_dist < topResults.top().first) {
                candidates.emplace(neighbot_dist, neighbor_id);
                topResults.emplace(neighbot_dist, neighbor_id);
                if (topResults.size() > ef_construction)
                    topResults.pop();
            }
        }
    }

    /* 将结果从小到大排序 */
    std::vector<std::pair<float, int>> result;
    while (!topResults.empty()) {
        result.push_back(topResults.top());
        topResults.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

void HNSW::insert(uint64_t key, const std::vector<float> &vec) {
    Node newNode(key, vec, random_layer());
    int newNodeId = nodes.size();

    /* 如果是第一个节点 */
    if (nodes.size() == 0) {
        nodes.push_back(newNode);
        entry_point = 0;
        top_layer   = newNode.max_layer;
        return;
    }

    /* 如果是已经存在的节点 */
    for (const auto &node : nodes) {
        if (node.key == key) {
            erase(node.key, node.vec);
            break;
        }
    }

    /* 如果是已经删除的节点 */
    for (auto it = deleted_nodes.begin(); it != deleted_nodes.end(); ++it) {
        if (it->first == key && it->second == vec) {
            deleted_nodes.erase(it);
            break;
        }
    }

    /* 如果是新的节点 */
    nodes.push_back(newNode);

    /* 贪心地搜索到 max_layer */
    int ep = entry_point;
    for (int layer = top_layer; layer > newNode.max_layer; --layer) {
        ep = search_layer_greedy(vec, layer, ep);
    }

    /* 搜索 0~max_layer 的邻居 */
    for (int layer = std::min(newNode.max_layer, top_layer); layer >= 0; --layer) {
        auto candidates = search_layer(vec, layer, ep);

        /* 选出前M个最近邻并连接 */
        size_t M = std::min(static_cast<size_t>(M), candidates.size());
        for (size_t i = 0; i < M; ++i) {
            int neighbor_id = candidates[i].second;
            connect(newNodeId, neighbor_id, layer);
        }
    }

    /* 更新 top_layer 和 ep */
    if (newNode.max_layer > top_layer) {
        entry_point = newNodeId;
        top_layer   = newNode.max_layer;
    }
}

void HNSW::erase(uint64_t key, const std::vector<float> &vec) {
    deleted_nodes.emplace_back(key, vec);
}

std::vector<uint64_t> HNSW::query(const std::vector<float> &q, int k) {
    /* 通过贪心的方式搜到第1层 */
    int ep = entry_point;
    for (int layer = top_layer; layer > 0; --layer) {
        ep = search_layer_greedy(q, layer, ep);
    }

    /* 在第0层进行精确搜索 */
    auto candidates = search_layer(q, 0, ep);

    /* 选出前k个最近邻 */
    std::vector<uint64_t> results;
    size_t size = std::min(static_cast<size_t>(k), candidates.size());
    for (size_t i = 0; i < size; ++i) {
        results.push_back(nodes[candidates[i].second].key);
    }

    return results;
}

void HNSW::putFile(const std::string &root) {
    /* if exit, delete */
    if (utils::dirExists(root)) {
        std::vector<std::string> files;
        utils::scanDir(root, files);
        for (auto &file : files) {
            std::string filename = root + "/" + file;
            utils::rmfile(filename.data());
        }
    }

    if (!utils::dirExists(root)) {
        utils::mkdir(root.data());
    }

    put_file_header(root);
    put_file_deleted_nodes(root);
    put_file_nodes(root);
}

void HNSW::loadFile(const std::string &root) {
    uint32_t size;    /* size if nodes */
    uint32_t dim;     /* dim of vec */
    load_file_header(root, size, dim);
    load_file_deleted_nodes(root, dim);
    load_file_nodes(root, size, dim);

    /* 删除 root 下所有文件 */
    std::vector<std::string> files;
    utils::scanDir(root, files);
    for (auto &file : files) {
        std::string filename = root + "/" + file;
        utils::rmfile(filename.data());
    }
}

/**
 * global header 的结构如下
 * uint32_t M
 * uint32_t M_max
 * uint32_t ef_construction
 * uint32_t m_L
 * uint32_t top_layer
 * uint32_t nodes.size()
 * uint32_t dim
 */
void HNSW::put_file_header(const std::string &root) {
    std::string filename = root + "/global_header.bin";
    std::ofstream output(filename, std::ios::binary);

    output.write(reinterpret_cast<const char *>(&M), sizeof(uint32_t));
    output.write(reinterpret_cast<const char *>(&M_max), sizeof(uint32_t));
    output.write(reinterpret_cast<const char *>(&ef_construction), sizeof(uint32_t));
    output.write(reinterpret_cast<const char *>(&m_L), sizeof(uint32_t));
    output.write(reinterpret_cast<const char *>(&top_layer), sizeof(uint32_t));
    uint32_t size = nodes.size();
    output.write(reinterpret_cast<const char *>(&size), sizeof(uint32_t));
    uint32_t dim = nodes.empty() ? 0 : nodes[0].vec.size();
    output.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
    output.close();
}

/**
 * deleted_nodes.bin 的结构如下
 * uint64_t key
 * float vec[dim]
 */
void HNSW::put_file_deleted_nodes(const std::string &root) {
    std::string filename = root + "/deleted_nodes.bin";
    std::ofstream output(filename, std::ios::binary);

    uint32_t dim = nodes.empty() ? 0 : nodes[0].vec.size();
    for (const auto &deleted_node : deleted_nodes) {
        output.write(reinterpret_cast<const char *>(&deleted_node.first), sizeof(uint64_t));
        output.write(reinterpret_cast<const char *>(deleted_node.second.data()), dim * sizeof(float));
    }
    output.close();
}

/**
 * nodes 结构如下
 * ── 0/              # 节点0的数据
 *    ├── header.bin  # 节点参数文件
 *    └── edges/      # 邻接表目录
 *        ├── 0.bin   # 第0层邻接表
 *        ├── 1.bin   # 第1层邻接表
 *        └── ...     # 其他存在的层级
 */
void HNSW::put_file_nodes(const std::string &root) {
    std::string nodes_dir = root + "/nodes";
    if (!utils::dirExists(nodes_dir)) {
        utils::mkdir(nodes_dir.data());
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        std::string node_dir = nodes_dir + "/" + std::to_string(i);
        if (!utils::dirExists(node_dir)) {
            utils::mkdir(node_dir.data());
        }

        /**
         * 写入 header.bin
         * uint32_t max_layer
         * uint64_t key
         * float vec[dim]
         */
        {
            std::string filename = node_dir + "/header.bin";
            std::ofstream output(filename, std::ios::binary);

            uint32_t max_layer = nodes[i].max_layer;
            output.write(reinterpret_cast<const char *>(&max_layer), sizeof(uint32_t));
            output.write(reinterpret_cast<const char *>(&nodes[i].key), sizeof(uint64_t));
            output.write(reinterpret_cast<const char *>(nodes[i].vec.data()), nodes[i].vec.size() * sizeof(float));
            output.close();
        }

        /* 写入 edges */
        {
            std::string edges_dir = node_dir + "/edges";
            if (!utils::dirExists(edges_dir)) {
                utils::mkdir(edges_dir.data());
            }

            /* 写入每一层的邻接表 */
            uint32_t max_layer = nodes[i].max_layer;
            for (uint32_t layer = 0; layer <= max_layer; ++layer) {
                std::string filename = edges_dir + "/" + std::to_string(layer) + ".bin";
                std::ofstream output(filename, std::ios::binary);

                /**
                 * 写入邻接表
                 * uint32_t num_neighbors
                 * uint32_t neighbors[num_neighbors]
                 */
                const auto &neighbors  = nodes[i].neighbors[layer];
                uint32_t num_neighbors = neighbors.size();
                output.write(reinterpret_cast<const char *>(&num_neighbors), sizeof(uint32_t));
                output.write(reinterpret_cast<const char *>(neighbors.data()), num_neighbors * sizeof(uint32_t));
                output.close();
            }
        }
    }
}

void HNSW::load_file_header(const std::string &root, uint32_t& size, uint32_t& dim) {
    std::string filename = root + "/global_header.bin";
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char *>(&M), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&M_max), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&ef_construction), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&m_L), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&top_layer), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&size), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

    input.close();
}

void HNSW::load_file_deleted_nodes(const std::string &root, const uint32_t& dim) {
    std::string filename = root + "/deleted_nodes.bin";
    std::ifstream input(filename, std::ios::binary);

    while (input.peek() != EOF) {
        uint64_t key;
        input.read(reinterpret_cast<char *>(&key), sizeof(uint64_t));

        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));

        deleted_nodes.emplace_back(key, vec);
    }
}

void HNSW::load_file_nodes(const std::string &root, const uint32_t &size, const uint32_t &dim) {
    std::string nodes_dir = root + "/nodes";

    /* 读取各个节点 */
    for (uint32_t i = 0; i < size; ++i) {
        std::string node_dir = nodes_dir + "/" + std::to_string(i);

        uint32_t max_layer;
        uint64_t key;
        std::vector<float> vec(dim);

        /* 读取 header.bin */
        {
            std::string filename = node_dir + "/header.bin";
            std::ifstream input(filename, std::ios::binary);

            input.read(reinterpret_cast<char *>(&max_layer), sizeof(uint32_t));
            input.read(reinterpret_cast<char *>(&key), sizeof(uint64_t));
            input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));

            nodes.emplace_back(key, vec, max_layer);
            input.close();
        }

        Node newNode(key, vec, max_layer);

        /* 读取 edges */
        {
            std::string edges_dir = node_dir + "/edges";

            /* 读取每一层的邻接表 */
            for (uint32_t layer = 0; layer <= max_layer; ++layer) {
                std::string filename = edges_dir + "/" + std::to_string(layer) + ".bin";
                std::ifstream input(filename, std::ios::binary);

                /* 读取邻接表 */
                uint32_t num_neighbors;
                input.read(reinterpret_cast<char *>(&num_neighbors), sizeof(uint32_t));

                std::vector<uint32_t> neighbors(num_neighbors);
                input.read(reinterpret_cast<char *>(neighbors.data()), num_neighbors * sizeof(uint32_t));

                newNode.neighbors[layer] = neighbors;
                input.close();
            }
        }

        /* 将节点添加到 HNSW 中 */
        nodes.push_back(newNode);
    }
}
