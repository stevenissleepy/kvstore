#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <set>
#include <cstdlib> 
#include <iostream> 

// HNSW图节点结构
struct HNSWNode {
    uint64_t key;             // 节点唯一标识符
    std::vector<float> vec;   // 节点向量数据
    std::vector<std::vector<int>> layers;  // 多层邻居结构
    int max_level;            // 节点所在最高层
};

class HNSWGraph {
private:
    std::vector<HNSWNode> nodes;   // 存储所有节点
    int entry_point = -1;          // 图的入口点
    int max_level = 0;             // 图的最大层数
    
    // HNSW参数
    const int M = 4;               // 每个节点的邻居数量
    const int Mmax = 6;            // 构建时允许的最大邻居数
    const int efConstruction = 40; // 构建索引时使用的搜索宽度
    const float mL = 1.0/log(6.0); // 用于随机层级生成的参数

    // 计算两个向量间的余弦距离
    float distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            return 2.0; // 不同维度向量返回最大距离
        }
        
        float dot = 0, norm_a = 0, norm_b = 0;
        for(size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a <= 0 || norm_b <= 0) {
            return 2.0; // 零向量返回最大距离
        }
        
        float similarity = dot / (sqrt(norm_a) * sqrt(norm_b));
        
        if (std::isnan(similarity) || std::isinf(similarity)) {
            return 2.0; // 无效结果返回最大距离
        }
        
        return 1 - similarity; // 转换为距离值
    }

    // 贪婪搜索算法，用于高层的快速导航
    int searchLayerGreedy(const std::vector<float>& q, int level, int entry_point_id) {
        if (nodes.empty() || entry_point_id == -1) 
            return -1;
            
        float dist_best = distance(q, nodes[entry_point_id].vec);
        int curr_node_id = entry_point_id;
        bool changed;
        
        // 贪婪搜索：持续移动到更近的邻居
        do {
            changed = false;
            
            for (int neighbor_id : nodes[curr_node_id].layers[level]) {
                float dist_neighbor = distance(q, nodes[neighbor_id].vec);
                
                if (dist_neighbor < dist_best) {
                    dist_best = dist_neighbor;
                    curr_node_id = neighbor_id;
                    changed = true;
                    break;  // 找到更近邻居立即移动
                }
            }
        } while (changed);
        
        return curr_node_id;
    }

    // 底层BFS搜索算法，用于精确查找最近邻
    std::vector<std::pair<float, int>> searchLayer(const std::vector<float>& q, int ef, int level, int entry_point_id = -1) {
        if (entry_point_id == -1) entry_point_id = entry_point;
        
        if (nodes.empty() || entry_point_id == -1) 
            return std::vector<std::pair<float, int>>();
        
        std::priority_queue<std::pair<float, int>> candidates;  // 候选点优先队列(负距离)
        std::priority_queue<std::pair<float, int>> topResults;  // 结果集优先队列(正距离)
        std::set<int> visited;                                  // 访问标记集合
        
        // 初始化搜索
        float d = distance(q, nodes[entry_point_id].vec);
        candidates.emplace(-d, entry_point_id);
        topResults.emplace(d, entry_point_id);
        visited.insert(entry_point_id);
        
        // BFS搜索过程
        while (!candidates.empty()) {
            auto current = candidates.top();
            float dist_current = -current.first;
            candidates.pop();
            
            // 剪枝优化
            if (topResults.size() >= ef && dist_current > topResults.top().first) {
                continue;
            }
            
            int current_id = current.second;
            
            // 扩展邻居
            for (int neighbor : nodes[current_id].layers[level]) {
                if (visited.count(neighbor)) continue;
                visited.insert(neighbor);
                
                float d_neighbor = distance(q, nodes[neighbor].vec);
                
                // 更新结果集
                if (topResults.size() < ef || d_neighbor < topResults.top().first) {
                    candidates.emplace(-d_neighbor, neighbor);
                    topResults.emplace(d_neighbor, neighbor);
                    
                    if (topResults.size() > ef) {
                        topResults.pop();
                    }
                }
            }
        }
        
        // 转换结果格式
        std::vector<std::pair<float, int>> result;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, 
                        std::greater<std::pair<float, int>>> minHeap;
        
        while (!topResults.empty()) {
            minHeap.push(topResults.top());
            topResults.pop();
        }
        while (!minHeap.empty()) {
            result.push_back(minHeap.top());
            minHeap.pop();
        }
        return result;
    }

public:
    // 插入新节点到图中
    void insert(uint64_t key, const std::vector<float>& vec) {
        // 验证向量有效性
        for (float val : vec) {
            if (std::isnan(val) || std::isinf(val)) {
                std::cerr << "Invalid vector values detected, skipping insert" << std::endl;
                return;
            }
        }
    
        // 创建新节点
        HNSWNode newNode;
        newNode.key = key;
        newNode.vec = vec;
        newNode.max_level = floor(-log((rand() / (float)RAND_MAX)) * mL); // 随机生成层级
        newNode.layers.resize(newNode.max_level + 1);
    
        // 处理空图情况
        if (nodes.empty()) {
            nodes.push_back(newNode);
            entry_point = 0;
            max_level = newNode.max_level;
            return;
        }
    
        // 分层查找插入位置
        int curr_ep = entry_point;
        for (int level = max_level; level > newNode.max_level; --level) {
            curr_ep = searchLayerGreedy(vec, level, curr_ep);
            if (curr_ep == -1) break;
        }
    
        // 添加节点
        int new_node_id = nodes.size();
        nodes.push_back(newNode);
    
        // 从高到低逐层建立连接
        for (int level = std::min(newNode.max_level, max_level); level >= 0; --level) {
            // 获取候选邻居
            auto candidates_with_dist = searchLayer(vec, efConstruction, level, curr_ep);
            
            std::vector<int> candidates;
            for (const auto& pair : candidates_with_dist) {
                candidates.push_back(pair.second);
            }
            
            if (!candidates.empty()) {
                curr_ep = candidates[0];
            }
            
            // 选择最优邻居集合
            std::vector<int> selected_neighbors;
            
            if (!candidates.empty()) {
                selected_neighbors.push_back(candidates[0]); // 最近的点总是被选择
            }
            
            // 启发式选择其他邻居
            for (size_t i = 1; i < candidates.size() && selected_neighbors.size() < M; ++i) {
                int candidate_id = candidates[i];
                bool add_candidate = true;
                
                // 确保选择分布均匀的邻居点
                for (int existing : selected_neighbors) {
                    float dist_existing_to_candidate = distance(nodes[existing].vec, nodes[candidate_id].vec);
                    float dist_query_to_candidate = distance(vec, nodes[candidate_id].vec);
                    
                    if (dist_existing_to_candidate < dist_query_to_candidate) {
                        add_candidate = false;
                        break;
                    }
                }
                
                if (add_candidate) {
                    selected_neighbors.push_back(candidate_id);
                }
            }
            
            // 补充邻居数量
            if (selected_neighbors.size() < M && candidates.size() > selected_neighbors.size()) {
                for (size_t i = 0; i < candidates.size() && selected_neighbors.size() < M; ++i) {
                    if (std::find(selected_neighbors.begin(), selected_neighbors.end(), candidates[i]) == selected_neighbors.end()) {
                        selected_neighbors.push_back(candidates[i]);
                    }
                }
            }
            
            // 建立连接并修剪超出部分
            for (int neighbor_id : selected_neighbors) {
                // 双向连接
                nodes[new_node_id].layers[level].push_back(neighbor_id);
                nodes[neighbor_id].layers[level].push_back(new_node_id);
                
                // 邻居修剪
                if (nodes[neighbor_id].layers[level].size() > Mmax) {
                    std::vector<std::pair<float, int>> dist_list;
                    for (int n : nodes[neighbor_id].layers[level]) {
                        float d = distance(nodes[n].vec, nodes[neighbor_id].vec);
                        dist_list.emplace_back(d, n);
                    }
                    sort(dist_list.begin(), dist_list.end());
                    nodes[neighbor_id].layers[level].clear();
                    for (int j = 0; j < std::min(Mmax, (int)dist_list.size()); ++j) {
                        nodes[neighbor_id].layers[level].push_back(dist_list[j].second);
                    }
                }
            }
        }
    
        // 更新入口点
        if (newNode.max_level > max_level) {
            entry_point = new_node_id;
            max_level = newNode.max_level;
        }
    }

    // 查询K近邻
    std::vector<std::pair<uint64_t, float>> query(const std::vector<float>& q, int k) {
        std::vector<std::pair<uint64_t, float>> results;
        if (nodes.empty()) return results;

        // 从高层开始逐层下降搜索
        int curr_ep = entry_point;
        for (int level = max_level; level > 0; --level) {
            curr_ep = searchLayerGreedy(q, level, curr_ep);
            if (curr_ep == -1) return results;
        }

        // 底层精确搜索
        auto candidates_with_dist = searchLayer(q, std::max(k, efConstruction), 0, curr_ep);
        
        // 转换结果格式
        for (int i = 0; i < std::min(k, (int)candidates_with_dist.size()); ++i) {
            results.emplace_back(nodes[candidates_with_dist[i].second].key, 
                            1 - candidates_with_dist[i].first);  // 距离转相似度
        }
        return results;
    }
};