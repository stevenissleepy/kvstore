cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 将 kvstore 编译成一个 lib
add_library(kvstore STATIC kvstore.cpp sstable.cpp sstablehead.cpp)
target_link_libraries(kvstore PUBLIC embedding skiplist bloom hnsw kvecTable)
target_include_directories(kvstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 添加子目录
add_subdirectory(third_party/llama.cpp)
add_subdirectory(lib)
add_subdirectory(test)