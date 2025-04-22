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
    int layer           = random_level();
    std::vector<int> ep = {enter_point};
    for (int l = max_layers - 1; l >= 0; l--) {
        // 搜索当前层的最近邻
        auto W = search_layer(vec, ep, 1, l);

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
        ep = W[0];
    }

    // 在底层搜索
    std::vector<int> result = search_layer(q, ep, ef_construction, 0);
    return result;
}

int HNSW::random_level() {
    std::exponential_distribution<double> exp(1.0);
    return std::min((int)-exp(rng), max_layers - 1);
}

std::vector<int> HNSW::search_layer(const std::vector<float> &q, int ep, int ef, int layer) {
    using HeapItem = std::pair<float, int>;
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>> candidates;
    std::unordered_map<int, bool> visited;

    // 初始化候选集
    for (int node_id : ep) {
        float dist = euclidean_distance(q, nodes[node_id].vec);
        candidates.push({dist, node_id});
        visited[node_id] = true;
    }

    // 扩展搜索
    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        for (int neighbor : nodes[current.second].neighbors[layer]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                float dist        = euclidean_distance(q, nodes[neighbor].vec);
                if (candidates.size() < ef || dist < candidates.top().first) {
                    candidates.push({dist, neighbor});
                    if (candidates.size() > ef)
                        candidates.pop();
                }
            }
        }
    }

    // 提取结果
    std::vector<int> result;
    while (!candidates.empty()) {
        result.push_back(candidates.top().second);
        candidates.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

void HNSW::connect(int a, int b, int layer) {
    auto &neighbors = nodes[a].neighbors[layer];
    if (std::find(neighbors.begin(), neighbors.end(), b) == neighbors.end()) {
        neighbors.push_back(b);
        if (neighbors.size() > M) {
            // 保留距离最近的M个邻居
            // （此处简化处理，实际需要按距离排序）
            neighbors.resize(M);
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
