#include "kvecTable.h"

#include "utils/utils.h"

#include <algorithm>
#include <fstream>
#include <limits>

kvecTable::kvecTable() {}

/* get 应该从后往前倒着搜索 */
std::vector<float> kvecTable::get(uint64_t key, const std::string &data_root) const {
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

void kvecTable::put(uint64_t key, const std::vector<float> &vec) {
    /* if is the first k-vec pair */
    if(dim == 0)
        dim = vec.size();

    keyTable.insert(key);
    table.emplace_back(key, vec);
}

void kvecTable::del(uint64_t key) {
    if (dim == 0)
        return;

    keyTable.erase(key);
    table.emplace_back(key, del_vec());
}

void kvecTable::putFile(const std::string &data_root) {
    if (dim == 0)
        return;

    /* create directory */
    if (!utils::dirExists(data_root)) {
        utils::mkdir(data_root.c_str());
    }

    /* get file name */
    int file_num = 0;
    std::vector<std::string> files;
    utils::scanDir(data_root, files);
    for (const auto &file : files) {
        int num = std::stoi(file.substr(0, file.find('.')));
        file_num = std::max(file_num, num);
    }
    std::string file_name = data_root + "/" + std::to_string(file_num + 1) + ".kvec";
    std::ofstream outfile(file_name, std::ios::binary);

    /* write dim */
    outfile.write(reinterpret_cast<const char *>(&dim), sizeof(uint64_t));

    /* write key-vec pair*/
    for (const auto &pair : table) {
        outfile.write(reinterpret_cast<const char *>(&pair.first), sizeof(uint64_t));
        outfile.write(reinterpret_cast<const char *>(pair.second.data()), dim * sizeof(float));
    }

    outfile.close();
}

void kvecTable::loadFile(const std::string &data_root) {
    std::vector<std::string> files;
    utils::scanDir(data_root, files);

    std::sort(files.begin(), files.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a.substr(0, a.find('.'))) < std::stoi(b.substr(0, b.find('.')));
    });

    for (const auto &file : files) {
        auto vecs = read_file(data_root + "/" + file);
        for (const auto &pair : vecs) {
            if(is_del_vec(pair.second)) {
                del(pair.first);
            } else {
                put(pair.first, pair.second);
            }
        }
    }
}

void kvecTable::reset(const std::string &data_root) {
    dim = 0;
    table.clear();
    if(utils::dirExists(data_root)) {
        utils::rmdir(data_root.c_str());
    }
}

std::unordered_set<uint64_t> kvecTable::getKeys() const {
    return keyTable;
}

std::vector<std::pair<uint64_t, std::vector<float>>> kvecTable::read_file(const std::string &file) const {
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

std::vector<float> kvecTable::del_vec() const {
    std::vector<float> result(dim, std::numeric_limits<float>::max());
    return result;
}

bool kvecTable::is_del_vec(const std::vector<float> &vec) const {
    for(const auto &v : vec) {
        if (v != std::numeric_limits<float>::max()) {
            return false;
        }
    }
    return true;
}
