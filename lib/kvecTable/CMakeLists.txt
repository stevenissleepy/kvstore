add_library(kvecTable kvecTable.cpp)

target_include_directories(kvecTable 
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/utils
)