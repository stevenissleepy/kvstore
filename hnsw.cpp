#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>

HNSW::HNSW(int m, int mmax, int ef, float mlParam) : M(m), Mmax(mmax), ef_construction(ef), mL(mlParam) {}

float HNSW::distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return 2.0f;
    float dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a <= 0 || norm_b <= 0)
        return 2.0f;
    float similarity = dot / (sqrt(norm_a) * sqrt(norm_b));
    return 1 - similarity;
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
    std::vector<std::pair<float, int>> results;
    std::priority_queue<std::pair<float, int>> candidates;
    std::priority_queue<std::pair<float, int>> topResults;
    std::set<int> visited;
    ep = ep == -1 ? ep: entry_point;

    float d = distance(q, nodes[ep].vec);
    candidates.emplace(-d, ep);
    topResults.emplace(d, ep);
    visited.insert(ep);

    while (!candidates.empty()) {
        auto current       = candidates.top();
        float dist_current = -current.first;
        candidates.pop();

        if (topResults.size() >= ef_construction && dist_current > topResults.top().first)
            continue;

        int current_id = current.second;
        for (int neighbor : nodes[current_id].neighbors[layer]) {
            if (visited.count(neighbor))
                continue;
            visited.insert(neighbor);
            float d_neighbor = distance(q, nodes[neighbor].vec);

            if (topResults.size() < ef_construction || d_neighbor < topResults.top().first) {
                candidates.emplace(-d_neighbor, neighbor);
                topResults.emplace(d_neighbor, neighbor);
                if (topResults.size() > ef_construction)
                    topResults.pop();
            }
        }
    }

    while (!topResults.empty()) {
        results.emplace_back(topResults.top().first, topResults.top().second);
        topResults.pop();
    }
    std::sort(results.begin(), results.end());
    return results;
}

void HNSW::insert(uint64_t key, const std::vector<float> &vec) {
    Node newNode;
    newNode.key       = key;
    newNode.vec       = vec;
    newNode.max_level = static_cast<int>(-log((rand() / (float)RAND_MAX)) * mL);
    newNode.neighbors.resize(newNode.max_level + 1);

    if (nodes.empty()) {
        nodes.push_back(newNode);
        entry_point = 0;
        max_level   = newNode.max_level;
        return;
    }

    int curr_ep = entry_point;
    for (int level = max_level; level > newNode.max_level; --level) {
        curr_ep = search_layer_greedy(vec, level, curr_ep);
    }

    int new_node_id = nodes.size();
    nodes.push_back(newNode);

    for (int level = std::min(newNode.max_level, max_level); level >= 0; --level) {
        auto candidates = search_layer(vec, level, curr_ep);
        std::vector<int> selected;
        for (const auto &pair : candidates) {
            if (selected.size() < M)
                selected.push_back(pair.second);
        }

        for (int neighbor : selected) {
            nodes[new_node_id].neighbors[level].push_back(neighbor);
            nodes[neighbor].neighbors[level].push_back(new_node_id);
        }
    }

    if (newNode.max_level > max_level) {
        entry_point = new_node_id;
        max_level   = newNode.max_level;
    }
}

std::vector<std::pair<uint64_t, float>> HNSW::query(const std::vector<float> &q, int k) {
    std::vector<std::pair<uint64_t, float>> results;
    if (nodes.empty())
        return results;

    int curr_ep = entry_point;
    for (int level = max_level; level > 0; --level) {
        curr_ep = search_layer_greedy(q, level, curr_ep);
    }

    auto candidates = search_layer(q, 0, curr_ep);
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), candidates.size()); ++i) {
        results.emplace_back(nodes[candidates[i].second].key, 1 - candidates[i].first);
    }
    return results;
}
