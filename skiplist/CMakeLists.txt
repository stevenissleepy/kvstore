add_library(skiplist STATIC skiplist.cpp)

target_include_directories(skiplist INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})