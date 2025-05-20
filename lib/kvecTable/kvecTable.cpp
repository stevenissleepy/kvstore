#include "kvecTable.h"

#include "utils/utils.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <thread>

KvecTable::KvecTable() {}

/* get 应该从后往前倒着搜索 */
std::vector<float> KvecTable::get(uint64_t key, const std::string &data_root) const {
    /* find in key table */
    if (keyTable.find(key) == keyTable.end()) 
        return del_vec();

    /* find in memory */
    for (auto it = table.rbegin(); it != table.rend(); ++it) {
        if (it->first == key) {
            return it->second;
        }
    }

    /* find in disk */
    std::vector<std::string> files;
    utils::scanDir(data_root, files);

    std::sort(files.begin(), files.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a.substr(0, a.find('.'))) > std::stoi(b.substr(0, b.find('.')));
    });

    for (const auto &file : files) {
        auto vecs = read_file(data_root + "/" + file);
        for (auto it = vecs.rbegin(); it != vecs.rend(); ++it) {
            if (it->first == key) {
                return it->second;
            }
        }
    }

    /* not found */
    return del_vec();
}

void KvecTable::put(uint64_t key, const std::vector<float> &vec) {
    /* if is the first k-vec pair */
    if(dim == 0)
        dim = vec.size();

    keyTable.insert(key);
    table.emplace_back(key, vec);
}

void KvecTable::del(uint64_t key) {
    if (dim == 0)
        return;

    keyTable.erase(key);
    table.emplace_back(key, del_vec());
}

void KvecTable::putFile(const std::string &data_root) {
    if (dim == 0 || table.empty())
        return;

    /* create directory */
    if (!utils::dirExists(data_root)) {
        utils::mkdir(data_root.c_str());
    }

    /* 获取当前已有的最大文件编号 */
    int file_num = 0;
    std::vector<std::string> files;
    utils::scanDir(data_root, files);
    for (const auto &file : files) {
        int num = std::stoi(file.substr(0, file.find('.')));
        file_num = std::max(file_num, num);
    }

    /* 分块并行写入 thread_num 个文件 */
    size_t total = table.size();
    size_t chunk_size = (total + thread_num - 1) / thread_num;
    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < thread_num; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, total);
        if (start >= end) break;
        std::string file_name = data_root + "/" + std::to_string(file_num + 1 + t) + ".kvec";
        threads.emplace_back([this, start, end, file_name]() {
            std::ofstream outfile(file_name, std::ios::binary);
            outfile.write(reinterpret_cast<const char *>(&dim), sizeof(uint64_t));
            for (size_t j = start; j < end; ++j) {
                const auto &pair = table[j];
                outfile.write(reinterpret_cast<const char *>(&pair.first), sizeof(uint64_t));
                outfile.write(reinterpret_cast<const char *>(pair.second.data()), dim * sizeof(float));
            }
            outfile.close();
        });
    }
    for (auto &th : threads) th.join();
}

void KvecTable::loadFile(const std::string &data_root) {
    /* if data_root don't exist*/
    if (!utils::dirExists(data_root)) {
        return;
    }

    /* load files */
    std::vector<std::string> files;
    utils::scanDir(data_root, files);

    std::sort(files.begin(), files.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a.substr(0, a.find('.'))) < std::stoi(b.substr(0, b.find('.')));
    });

    for (const auto &file : files) {
        const std::string file_name = data_root + "/" + file;
        /* read file */
        auto pairs = read_file(file_name);
        for (const auto &pair : pairs) {
            if(is_del_vec(pair.second)) {
                del(pair.first);
            } else {
                put(pair.first, pair.second);
            }
        }

        /* remove file */
        utils::rmfile(file_name.c_str()); 
    }
}

void KvecTable::reset(const std::string &data_root) {
    dim = 0;
    table.clear();
    if(utils::dirExists(data_root)) {
        std::vector<std::string> files;
        utils::scanDir(data_root, files);
        for (const auto &file : files) {
            utils::rmfile((data_root + "/" + file).c_str());
        }
    }
}

std::unordered_set<uint64_t> KvecTable::getKeys() const {
    return keyTable;
}

std::vector<std::pair<uint64_t, std::vector<float>>> KvecTable::read_file(const std::string &file) const {
    std::vector<std::pair<uint64_t, std::vector<float>>> result;

    std::ifstream infile(file, std::ios::binary);

    /* read dim */
    uint64_t dim;
    infile.read(reinterpret_cast<char *>(&dim), sizeof(uint64_t));

    /* read key-vec pair*/
    while (infile) {
        uint64_t key;
        std::vector<float> vector_data(dim);

        /* read key */
        infile.read(reinterpret_cast<char *>(&key), sizeof(uint64_t));
        if (infile.eof())
            break;

        /* read vec */
        infile.read(reinterpret_cast<char *>(vector_data.data()), dim * sizeof(float));
        if (infile.eof())
            break;

        result.emplace_back(key, std::move(vector_data));
    }

    infile.close();
    return result;
}

std::vector<float> KvecTable::del_vec() const {
    std::vector<float> result(dim, std::numeric_limits<float>::max());
    return result;
}

bool KvecTable::is_del_vec(const std::vector<float> &vec) const {
    for(const auto &v : vec) {
        if (v != std::numeric_limits<float>::max()) {
            return false;
        }
    }
    return true;
}
