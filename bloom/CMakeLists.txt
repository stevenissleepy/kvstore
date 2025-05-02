add_library(bloom STATIC bloom.cpp)

target_include_directories(bloom INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})