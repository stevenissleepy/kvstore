#include "hnsw.h"

// 构造函数：初始化参数和各层节点结构
HNSW::HNSW(int ml, int m, int ef) :
    max_layers(ml),
    M(m),
    ef_construction(ef),
    top_layer(-1),
    rng(std::random_device{}()) {
    layers.resize(max_layers); // 每层初始化为空
}

// 生成随机层数，指数衰减概率（0.5每层）
int HNSW::random_level() {
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    int level = 0;
    while (dist(rng) < 0.5 && level < max_layers - 1) {
        level++;
    }
    return level;
}

// 计算欧氏距离（带平方根）
float HNSW::euclidean_distance(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * 在指定层搜索最近邻
 *
 * @param q 查询向量
 * @param k 返回的邻居数量
 * @param ep 入口点
 * @param layer 当前层
 * @return 返回最近邻的节点ID
 */
std::vector<int> HNSW::search_layer(const std::vector<float> &q, int k, int ep, int layer) {
    std::unordered_map<int, Node> nodes = layers[layer];
    std::unordered_set<int> visited;
    auto cmp = [&](int a_id, int b_id) {
        return euclidean_distance(q, nodes[a_id].vec) > euclidean_distance(q, nodes[b_id].vec);
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> candidates(cmp);
    std::priority_queue<std::pair<float, int>> result;

    // 将起点放入候选队列和结果队列
    candidates.push(ep);
    visited.insert(ep);

    // 广搜+贪心
    while (!candidates.empty()) {
        int curr_id   = candidates.top();
        int curr_dist = euclidean_distance(q, nodes[curr_id].vec);
        candidates.pop();

        // 遍历当前节点的邻居
        for (int neighbor : nodes[curr_id].neighbors) {
            // 跳过已访问的节点
            if (visited.find(neighbor) != visited.end())
                continue;

            visited.insert(neighbor);
            float dist = euclidean_distance(q, nodes[neighbor].vec);
            if (dist < curr_dist) {
                candidates.push(neighbor);
            }
        }

        // 将 candidates 中的节点限制在 ef_construction 个
        if (candidates.size() > ef_construction) {
            candidates.pop();
        }
    }

    // 按距离从小到大排序结果
    std::vector<int> output;
    while (!result.empty()) {
        output.insert(output.begin(), result.top().second);
        result.pop();
    }
    return output;
}

// 连接两个节点，并修剪邻居到最多M个
void HNSW::connect(int a, int b, int layer) {
    auto &node_a = layers[layer][a];
    auto &node_b = layers[layer][b];

    // 将b加入a的邻居并排序修剪
    node_a.neighbors.push_back(b);
    std::sort(node_a.neighbors.begin(), node_a.neighbors.end(), [&](int x, int y) {
        return euclidean_distance(node_a.vec, layers[layer].at(x).vec) <
               euclidean_distance(node_a.vec, layers[layer].at(y).vec);
    });
    if (node_a.neighbors.size() > M) {
        node_a.neighbors.resize(M);
    }

    // 将a加入b的邻居并排序修剪（双向连接）
    node_b.neighbors.push_back(a);
    std::sort(node_b.neighbors.begin(), node_b.neighbors.end(), [&](int x, int y) {
        return euclidean_distance(node_b.vec, layers[layer].at(x).vec) <
               euclidean_distance(node_b.vec, layers[layer].at(y).vec);
    });
    if (node_b.neighbors.size() > M) {
        node_b.neighbors.resize(M);
    }
}

// 插入新节点到HNSW
void HNSW::insert(int id, const std::vector<float> &vec) {
    int layer = random_level();

    // 插入新节点
    for (int i = 0; i <= layer; ++i) {
        layers[i][id] = {id, vec, {}};
    }

    // 处理连接
    int ep = layers[top_layer].begin()->first;
    for (int i = top_layer; i >= 0; --i) {
        std::vector<int> neighbors = search_layer(vec, M, ep, i);
        ep                         = neighbors[0];

        // 连接新节点和邻居
        if (i <= layer) {
            for (int neighbor : neighbors) {
                connect(id, neighbor, i);
            }
        }
    }

    // 更新顶层
    top_layer = std::max(top_layer, layer);
}

// 查询最近的k个邻居
std::vector<int> HNSW::query(const std::vector<float> &q, int k) {
    // 查找顶层入口点
    int top_layer = max_layers - 1;
    while (top_layer >= 0 && layers[top_layer].empty()) {
        top_layer--;
    }
    if (top_layer < 0)
        return {};

    std::vector<int> eps = {layers[top_layer].begin()->first};

    // 从顶层到0层搜索
    for (int layer = top_layer; layer >= 0; --layer) {
        eps = search_layer(q, 1, eps[0], layer);
    }

    // 返回前k个结果
    if (eps.size() > k) {
        eps.resize(k);
    }
    return eps;
}
