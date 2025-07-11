#pragma once

#include "hnsw.h"
#include "kvecTable.h"
#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"

#include <map>
#include <set>

class KVStore : public KVStoreAPI {
private:
    /* compaction 工具函数 */
    bool sstable_num_out_of_limit(int level);
    void merge_sstables(std::vector<sstablehead> &ssts, std::map<uint64_t, std::string> &pairs);

private:
    // key-value
    skiplist *s = new skiplist(0.5);           // memtable
    std::vector<sstablehead> sstableIndex[15]; // the sshead for each level
    int totalLevel = -1;                       // 层数

    // key-vector
    KvecTable kvecTable; // memtable
    // HNSW hnsw;

public:
    KVStore(const std::string &dir);

    ~KVStore();

    void put(uint64_t key, const std::string &s) override;
    void put(uint64_t key, const std::vector<float> &vec);

    std::string get(uint64_t key) override;

    bool del(uint64_t key) override;

    void reset() override;

    void scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) override;

    void compaction();

    void delsstable(std::string filename);  // 从缓存中删除filename.sst， 并物理删除
    void addsstable(sstable ss, int level); // 将ss加入缓存

    std::string fetchString(std::string file, int startOffset, uint32_t len);

    void load_embedding_from_disk(const std::string &data_root="./data/embedding_data");
    // void save_hnsw_index_to_disk(const std::string &data_root="./data/hnsw_data");
    // void load_hnsw_index_from_disk(const std::string &data_root="./data/hnsw_data");

    std::vector<std::pair<std::uint64_t, std::string>> search_knn(std::string query, int k);
    std::vector<std::pair<std::uint64_t, std::string>> search_knn(std::vector<float> vec, int k);
    std::vector<std::pair<std::uint64_t, std::string>> search_knn_parallel(std::vector<float> vec, int k);
    // std::vector<std::pair<std::uint64_t, std::string>> search_knn_hnsw(std::string query, int k);
};
