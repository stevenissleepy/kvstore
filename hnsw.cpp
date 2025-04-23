#include "hnsw.h"

HNSW::HNSW(int ml, int m, int ef) : max_layers(ml), M(m), ef_construction(ef), enter_point(-1) {
    std::random_device rd;
    rng.seed(rd());
}

void HNSW::insert(int id, const std::vector<float> &vec) {
    Node new_node{id, vec, {}};
    nodes[id] = new_node;

    // 第一个节点
    if (enter_point == -1) {
        enter_point = id;
        return;
    }

    // 从最高层向下插入
    int layer = random_level();
    int ep    = enter_point;
    for (int l = max_layers - 1; l >= 0; l--) {
        // 搜索当前层的最近邻
        std::vector<int> W = search_layer(vec, ep, M, l);
        ep                 = W[0];

        // 连接新节点
        if (l <= layer) {
            for (int neighbor : W) {
                connect(id, neighbor, l);
                connect(neighbor, id, l);
            }
        }
    }
}

std::vector<int> HNSW::query(const std::vector<float> &q, int k) {

    // 从顶层向下搜索到倒数第二层
    int ep = enter_point;
    for (int l = max_layers - 1; l >= 1; l--) {
        std::vector<int> W = search_layer(q, ep, ef_construction, l);
        ep                 = W[0];
    }

    // 在底层搜索
    std::vector<int> result = search_layer(q, ep, k, 0);
    return result;
}

int HNSW::random_level() {
    std::exponential_distribution<double> exp(1.0);
    return std::min((int)-exp(rng), max_layers - 1);
}

std::vector<int> HNSW::search_layer(const std::vector<float> &q, int ep, int ef, int layer) {
    using Candidate = std::pair<float, int>;                                           // {距离, 节点ID}
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidates; // 最小堆，用于扩展
    std::priority_queue<Candidate> top_candidates;                                     // 最大堆，用于存储结果
    std::unordered_set<int> visited;                                                   // 记录访问过的节点

    // 初始化候选集
    float dist = euclidean_distance(q, nodes[ep].vec);
    candidates.push({dist, ep});
    top_candidates.push({dist, ep});
    visited.insert(ep);

    // 开始搜索
    while (!candidates.empty()) {
        auto [cur_dist, cur_node] = candidates.top();
        candidates.pop();

        // 遍历当前节点的邻居
        for (int neighbor : nodes[cur_node].neighbors[layer]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                float neighbor_dist = euclidean_distance(q, nodes[neighbor].vec);

                // 更新候选集和结果集
                if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                    candidates.push({neighbor_dist, neighbor});
                    top_candidates.push({neighbor_dist, neighbor});
                }

                // 如果结果集超过k个，移除最远的
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }

    // 提取结果
    std::vector<int> result;
    while (!top_candidates.empty()) {
        result.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(result.begin(), result.end()); // 按距离从小到大排序
    return result;
}

void HNSW::connect(int a, int b, int layer) {
    auto &neighbors = nodes[a].neighbors[layer];
    if (std::find(neighbors.begin(), neighbors.end(), b) == neighbors.end()) {
        neighbors.push_back(b);
        if (neighbors.size() > M) {
            std::priority_queue<std::pair<float, int>> neighbor_distances;
            for (int neighbor : neighbors) {
                float dist = euclidean_distance(nodes[a].vec, nodes[neighbor].vec);
                neighbor_distances.push({dist, neighbor});
            }

            neighbors.clear();
            while (neighbors.size() < M && !neighbor_distances.empty()) {
                neighbors.push_back(neighbor_distances.top().second);
                neighbor_distances.pop();
            }
        }
    }
}

float HNSW::euclidean_distance(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}
