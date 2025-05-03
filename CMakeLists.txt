cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 将 kvstore 编译成一个 lib
add_library(kvstore STATIC kvstore.cpp sstable.cpp sstablehead.cpp)
target_link_libraries(kvstore PUBLIC embedding skiplist bloom hnsw kvecTable)

# 定义一个函数，用于创建可执行目标
function(add_lsm_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PUBLIC kvstore)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# 添加可执行目标
add_lsm_executable(correctness correctness.cc)
add_lsm_executable(persistence persistence.cc)
add_lsm_executable(performance performance.cc)
add_lsm_executable(e2e_test "test/E2E_test.cpp")
add_lsm_executable(sim_test "test/sim_test.cpp")
add_lsm_executable(vp_test_1 "test/Vector_Persistent_Test_Phase1.cpp")
add_lsm_executable(vp_test_2 "test/Vector_Persistent_Test_Phase2.cpp")

# 添加子目录
add_subdirectory(third_party/llama.cpp)
add_subdirectory(lib)
add_subdirectory(test)