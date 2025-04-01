#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"

#include <map>
#include <set>

class KVStore : public KVStoreAPI {
private:
    // You can add your implementation here
    bool sstable_num_out_of_limit(int level);
    std::vector<float> str2vec(const std::string &s);

private:
    skiplist *s = new skiplist(0.5);            // memtable
    std::vector<sstablehead> sstableIndex[15];  // the sshead for each level
    int totalLevel = -1;                        // 层数

    std::map<uint64_t, std::vector<float>> kvecTable; // key-vector table

public:
    KVStore(const std::string &dir);

    ~KVStore();

    void put(uint64_t key, const std::string &s) override;

    std::string get(uint64_t key) override;

    bool del(uint64_t key) override;

    void reset() override;

    void scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) override;

    void compaction();

    void delsstable(std::string filename);  // 从缓存中删除filename.sst， 并物理删除
    void addsstable(sstable ss, int level); // 将ss加入缓存

    std::string fetchString(std::string file, int startOffset, uint32_t len);
};
