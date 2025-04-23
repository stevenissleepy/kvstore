#include "hnsw.h"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <queue>

using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::priority_queue;
using std::pair;

HNSW::HNSW(int ml, int m, int ef) :
    max_layers(ml),
    M(m),
    ef_construction(ef),
    top_layer(-1),
    rng(std::random_device{}()) {
    layers.resize(max_layers); // 每层初始化为空
}

int HNSW::random_level() {
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    int level = 0;
    while (dist(rng) < 0.5 && level < max_layers - 1) {
        level++;
    }
    return level;
}

float HNSW::euclidean_distance(const vector<float> &a, const vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float HNSW::similarity_cos(const vector<float> &a, const vector<float> &b) {
    int n1 = a.size();
    int n2 = b.size();
    assert(n1 == n2);
    int n = n1;

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

/**
 * 在指定层搜索最近邻
 *
 * @param q 查询向量
 * @param k 返回的邻居数量
 * @param ep 入口点
 * @param layer 当前层
 * @return 返回最近邻的节点ID
 */
vector<int> HNSW::search_layer(const vector<float> &q, int k, int ep, int layer) {
    unordered_map<int, Node> &nodes = layers[layer];
    unordered_set<int> visited;
    auto cmp = [&](int a_id, int b_id) {
        return similarity_cos(q, nodes[a_id].vec) < similarity_cos(q, nodes[b_id].vec);
    };
    priority_queue<int, vector<int>, decltype(cmp)> candidates(cmp);
    priority_queue<pair<float, int>> result;

    // 将起点放入候选队列和结果队列
    candidates.push(ep);
    visited.insert(ep);

    // 广搜+贪心
    while (!candidates.empty()) {
        int curr_id    = candidates.top();
        float curr_sim = similarity_cos(q, nodes[curr_id].vec);
        candidates.pop();

        // 检查当前节点是否够近
        if (result.size() < k) {
            result.push({curr_sim, curr_id});
        } else if (curr_sim > result.top().first) {
            result.pop();
            result.push({curr_sim, curr_id});
        }

        // 遍历当前节点的邻居
        for (int neighbor : nodes[curr_id].neighbors) {
            // 跳过已访问的节点
            if (visited.find(neighbor) != visited.end())
                continue;

            visited.insert(neighbor);
            float sim = similarity_cos(q, nodes[neighbor].vec);
            if (sim > curr_sim || result.size() < k) {
                candidates.push(neighbor);
            }
        }

        // 将 candidates 中的节点限制在 ef_construction 个
        if (candidates.size() > ef_construction) {
            candidates.pop();
        }
    }

    // 按相似度从大到小排序
    vector<int> output;
    while (!result.empty()) {
        output.insert(output.begin(), result.top().second);
        result.pop();
    }
    return output;
}

void HNSW::connect(int a, int b, int layer) {
    auto &node_a = layers[layer][a];
    auto &node_b = layers[layer][b];

    // 将b加入a的邻居并排序修剪
    node_a.neighbors.push_back(b);
    std::sort(node_a.neighbors.begin(), node_a.neighbors.end(), [&](int x, int y) {
        return similarity_cos(node_a.vec, layers[layer].at(x).vec) >
               similarity_cos(node_a.vec, layers[layer].at(y).vec);
    });
    if (node_a.neighbors.size() > M) {
        node_a.neighbors.resize(M);
    }

    // 将a加入b的邻居并排序修剪（双向连接）
    node_b.neighbors.push_back(a);
    std::sort(node_b.neighbors.begin(), node_b.neighbors.end(), [&](int x, int y) {
        return similarity_cos(node_b.vec, layers[layer].at(x).vec) >
               similarity_cos(node_b.vec, layers[layer].at(y).vec);
    });
    if (node_b.neighbors.size() > M) {
        node_b.neighbors.resize(M);
    }
}

void HNSW::insert(int id, const vector<float> &vec) {
    int layer = random_level();
    int ep    = top_layer == -1 ? -1 : layers[top_layer].begin()->first;

    /* 插入新节点 */
    for (int i = 0; i <= layer; ++i) {
        layers[i][id] = {id, vec, {}};
    }

    /* 处理连接 */
    for (int i = top_layer; i >= 0; --i) {
        vector<int> neighbors = search_layer(vec, M, ep, i);
        ep                         = neighbors[0];

        if (i <= layer) { // 连接新节点和邻居
            for (int neighbor : neighbors) {
                connect(id, neighbor, i);
            }
        }
    }

    /* 更新顶层 */
    top_layer = std::max(top_layer, layer);
}

vector<int> HNSW::query(const vector<float> &q, int k) {
    int ep = layers[top_layer].begin()->first;
    vector<int> neighbors;

    for (int i = top_layer; i >= 0; --i) {
        neighbors = search_layer(q, k, ep, i);
        ep        = neighbors[0];
    }

    return neighbors;
}

void HNSW::check_layers() {
    std::ofstream out("layers.txt");

    for (int i = top_layer; i >= 0; --i) {
        out << "Layer " << i << ":\n";

        // 遍历当前层的所有节点
        for (const auto &node_pair : layers[i]) {
            int node_id      = node_pair.first;
            const Node &node = node_pair.second;

            // 输出节点及其邻居
            out << "Node " << node_id << ": ";
            for (int neighbor : node.neighbors) {
                out << neighbor << " ";
            }
            out << "\n";
        }

        out << "\n";
    }

    out.close();
}
