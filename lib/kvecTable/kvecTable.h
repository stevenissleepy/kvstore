#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

class KvecTable {
public:
    KvecTable();

    std::vector<float> get(uint64_t key, const std::string &data_root = "./data/kvec") const;

    void put(uint64_t key, const std::vector<float> &vec);

    void del(uint64_t key);

    void putFile(const std::string &data_root = "./data/kvec");

    void loadFile(const std::string &data_root = "./data/kvec");

    void reset(const std::string &data_root = "./data/kvec");

    std::unordered_set<uint64_t> getKeys() const;

private:
    std::vector<std::pair<uint64_t, std::vector<float>>> read_file(const std::string &file) const;

    std::vector<float> del_vec() const;
    bool is_del_vec(const std::vector<float> &vec) const;

private:
    std::vector<std::pair<uint64_t, std::vector<float>>> table;
    std::unordered_set<uint64_t> keyTable;

    uint64_t dim = 0;
};
