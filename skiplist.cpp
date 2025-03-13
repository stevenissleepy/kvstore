#include "skiplist.h"

#include <vector>

double skiplist::my_rand() {
    return static_cast<double>(rand()) / RAND_MAX;
}

int skiplist::randLevel() {
    int level = 1;
    while (my_rand() < p && level < MAX_LEVEL)
        level++;
    return level;
}

void skiplist::insert(uint64_t key, const std::string &val) {
    std::vector<slnode *> update(MAX_LEVEL, nullptr);
    slnode *current = head;

    // 查找插入位置
    for (int i = curMaxL - 1; i >= 0; i--) {
        while (current->nxt[i] != tail && current->nxt[i]->key < key) {
            current = current->nxt[i];
        }
        update[i] = current;
    }

    // 检查是否已存在
    if (current->nxt[0] != tail && current->nxt[0]->key == key) {
        return;
    }

    // 生成新节点层数
    int newLevel = randLevel();
    if (newLevel > curMaxL) {
        for (int i = curMaxL; i < newLevel; i++) {
            update[i] = head;
        }
        curMaxL = newLevel;
    }

    // 创建新节点并更新指针
    slnode *newNode = new slnode(key, val, NORMAL);
    for (int i = 0; i < newLevel; i++) {
        newNode->nxt[i]   = update[i]->nxt[i];
        update[i]->nxt[i] = newNode;
    }
}

std::string skiplist::search(uint64_t key) {
    slnode *current = head;
    for (int i = curMaxL - 1; i >= 0; i--) {
        while (current->nxt[i] != tail && current->nxt[i]->key < key) {
            current = current->nxt[i];
        }
    }
    current = current->nxt[0];
    if (current != tail && current->key == key) {
        return current->val;
    }
    return "";
}

bool skiplist::del(uint64_t key, uint32_t len) {
    std::vector<slnode *> update(MAX_LEVEL, nullptr);
    slnode *current = head;

    // 查找目标节点
    for (int i = curMaxL - 1; i >= 0; i--) {
        while (current->nxt[i] != tail && current->nxt[i]->key < key) {
            current = current->nxt[i];
        }
        update[i] = current;
    }
    current = current->nxt[0];

    // 如果不存在
    if (current->type == TAIL || current->key != key) {
        return false;
    }

    // 更新前驱指针
    for (int i = 0; i < curMaxL; i++) {
        if (update[i]->nxt[i] != current) {
            break;
        }
        update[i]->nxt[i] = current->nxt[i];
    }

    // 更新当前层数
    while (curMaxL > 1 && !head->nxt[curMaxL - 1]) {
        curMaxL--;
    }

    delete current;
    return true;
}

void skiplist::scan(
    uint64_t key1,
    uint64_t key2,
    std::vector<std::pair<uint64_t, std::string>> &list
) {
    slnode *current = head;
    for (int i = curMaxL - 1; i >= 0; i--) {
        while (current->nxt[i] != tail && current->nxt[i]->key < key1) {
            current = current->nxt[i];
        }
    }
    current = current->nxt[0];
    while (current != tail && current->key <= key2) {
        list.push_back(std::make_pair(current->key, current->val));
        current = current->nxt[0];
    }
}

slnode *skiplist::lowerBound(uint64_t key) {
    slnode *current = head;
    for (int i = curMaxL - 1; i >= 0; i--) {
        while (current->nxt[i] != tail && current->nxt[i]->key < key) {
            current = current->nxt[i];
        }
    }
    return current->nxt[0];
}

void skiplist::reset() {
    slnode *current = head->nxt[0];
    while (current != tail) {
        slnode *tmp = current;
        current = current->nxt[0];
        delete tmp;
    }

    // 重置参数
    s = 1;
    bytes = 0x0;
    curMaxL = 1;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        head->nxt[i] = tail;
    }
}

uint32_t skiplist::getBytes() {
    return bytes;
}
