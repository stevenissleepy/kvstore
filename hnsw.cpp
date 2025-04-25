#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

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
    nodes[id].neighbors[layer].push_back(neighbor_id);
    nodes[neighbor_id].neighbors[layer].push_back(id);
}

/* 余弦相似度越大，距离越短 */
inline float HNSW::distance(const std::vector<float> &a, const std::vector<float> &b) {
    return - similarity_cos(a, b);
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

int HNSW::search_layer_greedy(const std::vector<float> &q, int layer, int ep) {
    int current_id     = ep;
    Node &current_node = nodes[current_id];
    int current_dist   = distance(q, current_node.vec);

    while (true) {
        bool found_closer = false;

        // 找出邻居中最近的节点
        for (int neighbor_id : current_node.neighbors[layer]) {
            float neighbor_dist = distance(q, nodes[neighbor_id].vec);

            if (neighbor_dist < current_dist) {
                current_id   = neighbor_id;
                current_dist = neighbor_dist;
            }
        }

        // 没有更近节点时终止循环
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
    nodes.push_back(newNode);

    /* 如果是第一个节点 */
    if (nodes.size() == 1) {
        nodes.push_back(newNode);
        entry_point = 0;
        top_layer   = newNode.max_layer;
        return;
    }

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

std::vector<std::pair<uint64_t, float>> HNSW::query(const std::vector<float> &q, int k) {
    /* 通过贪心的方式搜到第1层 */
    int ep = entry_point;
    for (int layer = top_layer; layer > 0; --layer) {
        ep = search_layer_greedy(q, layer, ep);
    }

    /* 在第0层进行精确搜索 */
    auto candidates = search_layer(q, 0, ep);
    
    /* 选出前k个最近邻 */
    std::vector<std::pair<uint64_t, float>> results;
    size_t size = std::min(static_cast<size_t>(k), candidates.size());
    for (size_t i = 0; i < size; ++i) {
        results.emplace_back(nodes[candidates[i].second].key, -candidates[i].first);
    }
    return results;
}
