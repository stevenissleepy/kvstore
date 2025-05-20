#include "kvstore.h"

#include "embedding.h"
#include "skiplist.h"
#include "sstable.h"
#include "utils.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>

static const std::string DEL = "~DELETED~";
const uint32_t MAXSIZE       = 2 * 1024 * 1024;

struct poi {
    int sstableId; // vector中第几个sstable
    int pos;       // 该sstable的第几个key-offset
    uint64_t time;
    Index index;
};

struct cmpPoi {
    bool operator()(const poi &a, const poi &b) {
        if (a.index.key == b.index.key)
            return a.time < b.time;
        return a.index.key > b.index.key;
    }
};

bool KVStore::sstable_num_out_of_limit(int level) {
    int limit = 1 << (level + 1); // 2^(k+1)
    return sstableIndex[level].size() > limit;
}

void KVStore::merge_sstables(std::vector<sstablehead> &ssts, std::map<uint64_t, std::string> &pairs) {
    for (sstablehead &it : ssts) {
        sstable ss(it);
        int cnt = ss.getCnt();
        for (int i = 0; i < cnt; ++i) {
            uint64_t key    = ss.getKey(i);
            std::string val = ss.getData(i);
            pairs[key]      = val;
        }
        delsstable(it.getFilename());
    }
}

void KVStore::load_embedding_from_disk(const std::string &data_root) {
    kvecTable.loadFile(data_root);
}

// void KVStore::save_hnsw_index_to_disk(const std::string &data_root) {
//     hnsw.putFile(data_root);
// }

// void KVStore::load_hnsw_index_from_disk(const std::string &data_root) {
//     hnsw.loadFile(data_root);
// }

KVStore::KVStore(const std::string &dir) :
    KVStoreAPI(dir) // read from sstables
{
    /* read k-value */
    for (totalLevel = 0;; ++totalLevel) {
        std::string path = dir + "/level-" + std::to_string(totalLevel) + "/";
        std::vector<std::string> files;
        if (!utils::dirExists(path)) {
            totalLevel--;
            break; // stop read
        }
        int nums = utils::scanDir(path, files);
        sstablehead cur;
        for (int i = 0; i < nums; ++i) {       // 读每一个文件头
            std::string url = path + files[i]; // url, 每一个文件名
            cur.loadFileHead(url.data());
            sstableIndex[totalLevel].push_back(cur);
            TIME = std::max(TIME, cur.getTime()); // 更新时间戳
        }
    }
}

KVStore::~KVStore() {
    /* put k-vec */
    kvecTable.putFile("./data/embedding_data");

    /* put k-value */
    sstable ss(s);
    if (!ss.getCnt())
        return; // empty sstable
    std::string path = std::string("./data/level-0/");
    if (!utils::dirExists(path)) {
        utils::mkdir(path.data());
        totalLevel = 0;
    }
    ss.putFile(ss.getFilename().data());
    compaction(); // 从0层开始尝试合并
}

/**
 * Insert/Update the key-value pair.
 * No return values for simplicity.
 */
void KVStore::put(uint64_t key, const std::string &val) {
    uint32_t nxtsize = s->getBytes();
    std::string res  = s->search(key);
    if (!res.length()) { // new add
        nxtsize += 12 + val.length();
    } else
        nxtsize = nxtsize - res.length() + val.length(); // change string
    if (nxtsize + 10240 + 32 <= MAXSIZE)
        s->insert(key, val); // 小于等于（不超过） 2MB
    else {
        /* put k-vec */
        kvecTable.putFile("./data/embedding_data");

        /* put k-value */
        sstable ss(s);
        s->reset();
        std::string url  = ss.getFilename();
        std::string path = "./data/level-0";
        if (!utils::dirExists(path)) {
            utils::mkdir(path.data());
            totalLevel = 0;
        }
        addsstable(ss, 0);      // 加入缓存
        ss.putFile(url.data()); // 加入磁盘
        compaction();
        s->insert(key, val);
    }
}

void KVStore::put(uint64_t key, const std::vector<float> &vec) {
    kvecTable.put(key, vec);
    // hnsw.insert(key, vec);
}

/**
 * Returns the (string) value of the given key.
 * An empty string indicates not found.
 */
std::string KVStore::get(uint64_t key) //
{
    uint64_t time = 0;
    int goalOffset;
    uint32_t goalLen;
    std::string goalUrl;
    std::string res = s->search(key);

    /* 在memtable中找到, 或者是deleted，说明最近被删除过 */
    if (res.length()) { 
        if (res == DEL)
            return "";
        return res;
    }

    /* 在sstable中寻找 */
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead& it : sstableIndex[level]) {
            if (key < it.getMinV() || key > it.getMaxV())
                continue;
            uint32_t len;
            int offset = it.searchOffset(key, len);
            if (offset == -1) {
                if (!level)
                    continue;
                else
                    break;
            }
            // sstable ss;
            // ss.loadFile(it.getFilename().data());
            if (it.getTime() > time) { // find the latest head
                time       = it.getTime();
                goalUrl    = it.getFilename();
                goalOffset = offset + 32 + 10240 + 12 * it.getCnt();
                goalLen    = len;
            }
        }
        if (time)
            break; // only a test for found
    }
    if (!goalUrl.length())
        return ""; // not found a sstable
    res = fetchString(goalUrl, goalOffset, goalLen);
    if (res == DEL)
        return "";
    return res;
}

/**
 * Delete the given key-value pair if it exists.
 * Returns false iff the key is not found.
 */
bool KVStore::del(uint64_t key) {
    /* del in lsm-tree */
    std::string res = get(key);
    if (!res.length())
        return false; // not exist
    put(key, DEL);    // put a del marker
    
    /* del in k-vec */
    // std::vector<float> vec = kvecTable.get(key);
    kvecTable.del(key);

    /* del in hnsw */
    // hnsw.erase(key, vec);

    return true;
}

/**
 * This resets the kvstore. All key-value pairs should be removed,
 * including memtable and all sstables files.
 */
void KVStore::reset() {
    s->reset(); // 先清空memtable
    std::vector<std::string> files;
    for (int level = 0; level <= totalLevel; ++level) { // 依层清空每一层的sstables
        std::string path = std::string("./data/level-") + std::to_string(level);
        int size         = utils::scanDir(path, files);
        for (int i = 0; i < size; ++i) {
            std::string file = path + "/" + files[i];
            utils::rmfile(file.data());
        }
        utils::rmdir(path.data());
        sstableIndex[level].clear();
    }
    totalLevel = -1;

    /* 清空 kvtable*/
    kvecTable.reset("./data/embedding_data");
}

/**
 * Return a list including all the key-value pair between key1 and key2.
 * keys in the list should be in an ascending order.
 * An empty string indicates not found.
 */

struct myPair {
    uint64_t key, time;
    int id, index;
    std::string filename;

    myPair(uint64_t key, uint64_t time, int index, int id,
           std::string file) { // construct function
        this->time     = time;
        this->key      = key;
        this->id       = id;
        this->index    = index;
        this->filename = file;
    }
};

struct cmp {
    bool operator()(myPair &a, myPair &b) {
        if (a.key == b.key)
            return a.time < b.time;
        return a.key > b.key;
    }
};

void KVStore::scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) {
    std::vector<std::pair<uint64_t, std::string>> mem;
    // std::set<myPair> heap; // 维护一个指针最小堆
    std::priority_queue<myPair, std::vector<myPair>, cmp> heap;
    // std::vector<sstable> ssts;
    std::vector<sstablehead> sshs;
    s->scan(key1, key2, mem);   // add in mem
    std::vector<int> head, end; // [head, end)
    int cnt = 0;
    if (mem.size())
        heap.push(myPair(mem[0].first, INF, 0, -1, "qwq"));
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            if (key1 > it.getMaxV() || key2 < it.getMinV())
                continue; // 无交集
            int hIndex = it.lowerBound(key1);
            int tIndex = it.lowerBound(key2);
            if (hIndex < it.getCnt()) { // 此sstable可用
                // sstable ss; // 读sstable
                std::string url = it.getFilename();
                // ss.loadFile(url.data());

                heap.push(myPair(it.getKey(hIndex), it.getTime(), hIndex, cnt++, url));
                head.push_back(hIndex);
                if (it.search(key2) == tIndex)
                    tIndex++; // tIndex为第一个不可的
                end.push_back(tIndex);
                // ssts.push_back(ss); // 加入ss
                sshs.push_back(it);
            }
        }
    }
    uint64_t lastKey = INF; // only choose the latest key
    while (!heap.empty()) { // 维护堆
        myPair cur = heap.top();
        heap.pop();
        if (cur.id >= 0) { // from sst
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                uint32_t start  = sshs[cur.id].getOffset(cur.index - 1);
                uint32_t len    = sshs[cur.id].getOffset(cur.index) - start;
                uint32_t scnt   = sshs[cur.id].getCnt();
                std::string res = fetchString(cur.filename, 10240 + 32 + scnt * 12 + start, len);
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, res);
            }
            if (cur.index + 1 < end[cur.id]) { // add next one to heap
                heap.push(myPair(sshs[cur.id].getKey(cur.index + 1), cur.time, cur.index + 1, cur.id, cur.filename));
            }
        } else { // from mem
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                std::string res = mem[cur.index].second;
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, mem[cur.index].second);
            }
            if (cur.index < mem.size() - 1) {
                heap.push(myPair(mem[cur.index + 1].first, cur.time, cur.index + 1, -1, cur.filename));
            }
        }
    }
}

void KVStore::compaction() {
    int curLevel = 0;
    // TODO here

    while (sstable_num_out_of_limit(curLevel)) {
        // 如果下一层的文件夹不存在，则创建
        std::string path = std::string("./data/level-") + std::to_string(curLevel + 1);
        if (!utils::dirExists(path)) {
            utils::mkdir(path.data());
        }

        // level-0 取3个 sstable
        // level-n 取超出限制的 sstable
        std::vector<sstablehead> ssts;
        int size = sstableIndex[curLevel].size();
        size -= (1 << (curLevel + 1));     // 多出来的 sstable 个数
        size = (curLevel == 0) ? 3 : size; // level-0 取3个 sstable
        for (int i = 0; i < size; ++i) {
            ssts.push_back(sstableIndex[curLevel][i]);
        }

        // 取 ssts 中的 key 区间
        uint64_t minKey = INF, maxKey = 0;
        for (sstablehead &it : ssts) {
            minKey = std::min(minKey, it.getMinV());
            maxKey = std::max(maxKey, it.getMaxV());
        }

        // 找出 level-(n+1) 中 key 值在区间内的 sstable
        for (sstablehead &it : sstableIndex[curLevel + 1]) {
            if (it.getMinV() <= maxKey && it.getMaxV() >= minKey) {
                ssts.push_back(it);
            }
        }

        // 将 ssts 中的 sstable 按时间戳排序，时间戳小的在前
        // 保证下一步中时间戳较大的 key 会覆盖时间戳较小的 key
        std::sort(ssts.begin(), ssts.end());
        uint64_t maxTime = ssts.back().getTime();

        // 合并 ssts 中的 sstable
        std::map<uint64_t, std::string> pairs;
        merge_sstables(ssts, pairs);

        // 生成新的 sstable
        sstable newSs;
        uint32_t maxNameSuffix = 0;
        for (sstablehead &it : sstableIndex[curLevel + 1]) {
            maxNameSuffix = it.getTime() == maxTime ? std::max(maxNameSuffix, it.getNameSuf()) : maxNameSuffix;
        }
        newSs.setTime(maxTime);             // 时间戳为 ssts 中最大的时间戳
        newSs.setNamesuffix(maxNameSuffix); // 保证文件名不会重复

        for (auto it : pairs) {
            if (newSs.checkSize(it.second, curLevel + 1, 0)) {
                addsstable(newSs, curLevel + 1);
                newSs.reset();
            }
            // 如果是最后一层，且 key 对应的 value 为 DEL，则不插入
            if (curLevel + 1 == totalLevel && it.second == DEL) {
                continue;
            }
            newSs.insert(it.first, it.second);
        }
        newSs.checkSize("", curLevel + 1, 1);
        addsstable(newSs, curLevel + 1);

        // 将 sstableIndex[curLevel+1] 排序
        std::sort(sstableIndex[curLevel + 1].begin(), sstableIndex[curLevel + 1].end());

        curLevel++;
    }

    // 更新 totalLevel
    totalLevel = std::max(totalLevel, curLevel);
}

void KVStore::delsstable(std::string filename) {
    for (int level = 0; level <= totalLevel; ++level) {
        int size = sstableIndex[level].size(), flag = 0;
        for (int i = 0; i < size; ++i) {
            if (sstableIndex[level][i].getFilename() == filename) {
                sstableIndex[level].erase(sstableIndex[level].begin() + i);
                flag = 1;
                break;
            }
        }
        if (flag)
            break;
    }
    int flag = utils::rmfile(filename.data());
    if (flag != 0) {
        std::cout << "delete fail!" << std::endl;
        std::cout << strerror(errno) << std::endl;
    }
}

void KVStore::addsstable(sstable ss, int level) {
    sstableIndex[level].push_back(ss.getHead());
}

char strBuf[2097152];

/**
 * @brief Fetches a substring from a file starting at a given offset.
 *
 * This function opens a file in binary read mode, seeks to the specified start offset,
 * reads a specified number of bytes into a buffer, and returns the buffer as a string.
 *
 * @param file The path to the file from which to read the substring.
 * @param startOffset The offset in the file from which to start reading.
 * @param len The number of bytes to read from the file.
 * @return A string containing the read bytes.
 */
std::string KVStore::fetchString(std::string file, int startOffset, uint32_t len) {
    // TODO here
    FILE *fp = fopen(file.data(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("open file failed");
    }
    fseek(fp, startOffset, SEEK_SET);
    fread(strBuf, 1, len, fp);
    fclose(fp);
    return std::string(strBuf, len);
}

std::vector<std::pair<std::uint64_t, std::string>> KVStore::search_knn(std::string query, int k) {
    std::vector<float> vec = embedding_single(query);

    /* 获取每个 kv 与目标的余弦相似度 */
    size_t n_embd = vec.size();
    std::vector<std::pair<uint64_t, float>> ksimTable;
    std::unordered_set<uint64_t> keys = kvecTable.getKeys();
    for (auto it : keys) {
        uint64_t key            = it;
        std::vector<float> vec2 = kvecTable.get(key);
        float sim               = common_embd_similarity_cos(vec.data(), vec2.data(), n_embd);
        ksimTable.emplace_back(key, sim);
    }

    /* 根据余弦相似度排序 */
    std::partial_sort(
        ksimTable.begin(),
        ksimTable.begin() + k,
        ksimTable.end(),
        [](const std::pair<uint64_t, float> &a, const std::pair<uint64_t, float> &b) { return a.second > b.second; }
    );

    /* 取前k个 */
    std::vector<std::pair<std::uint64_t, std::string>> res;
    for (int i = 0; i < k; ++i) {
        if(ksimTable.size() <= i) break;
        res.emplace_back(ksimTable[i].first, get(ksimTable[i].first));
    }

    return res;
}

// std::vector<std::pair<std::uint64_t, std::string>> KVStore::search_knn_hnsw(std::string query, int k) {
//     /* 找出最接近的 k 个 key */
//     std::vector<float> vec    = embedding_single(query);
//     std::vector<uint64_t> knn = hnsw.query(vec, k);

//     /* 通过 key 找到对应的 key-value */
//     std::vector<std::pair<std::uint64_t, std::string>> res;
//     for (uint64_t key : knn) {
//         res.emplace_back(key, get(key));
//     }
//     return res;
// }
