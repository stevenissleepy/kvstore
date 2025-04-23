cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 公共源文件和头文件
set(COMMON_SOURCES
    kvstore.cc
    skiplist.cpp
    sstable.cpp
    bloom.cpp
    sstablehead.cpp
)

# 定义一个函数，用于创建可执行目标
function(add_lsm_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${COMMON_SOURCES})
    target_link_libraries(${target_name} PUBLIC embedding)
endfunction()

# 添加可执行目标
add_lsm_executable(correctness correctness.cc)
add_lsm_executable(persistence persistence.cc)
add_lsm_executable(performance performance.cc)
add_lsm_executable(e2e_test "test/E2E_test.cpp")
add_lsm_executable(sim_test "test/sim_test.cpp")

# 添加子目录
add_subdirectory(third_party/llama.cpp)
add_subdirectory(embedding)
add_subdirectory(test)