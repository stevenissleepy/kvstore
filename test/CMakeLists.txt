# LSM Tree test
add_executable(correctness correctness.cc)
add_executable(persistence persistence.cc)
add_executable(performance performance.cc)
target_link_libraries(correctness PUBLIC kvstore)
target_link_libraries(persistence PUBLIC kvstore)
target_link_libraries(performance PUBLIC kvstore)

# embedding test
add_executable(Embedding_Test Embedding_Test.cpp)
target_link_libraries(Embedding_Test PUBLIC embedding)

# E2E test
add_executable(e2e_test E2E_test.cpp)
target_link_libraries(e2e_test PUBLIC kvstore)

# vector persistenct test
add_executable(vp_test_1 Vector_Persistent_Test_Phase1.cpp)
add_executable(vp_test_2 Vector_Persistent_Test_Phase2.cpp)
target_link_libraries(vp_test_1 PUBLIC kvstore)
target_link_libraries(vp_test_2 PUBLIC kvstore)

# HNSW delete test
add_executable(hnsw_delete_test HNSW_Delete_Test.cpp)
target_link_libraries(hnsw_delete_test PUBLIC kvstore)

# HNSW persistent test
add_executable(hp_test1 HNSW_Persistent_Test_Phase1.cpp)
add_executable(hp_test2 HNSW_Persistent_Test_Phase2.cpp)
target_link_libraries(hp_test1 PUBLIC kvstore)
target_link_libraries(hp_test2 PUBLIC kvstore)