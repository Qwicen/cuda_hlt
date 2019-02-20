include_directories(${CMAKE_SOURCE_DIR}/tests)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/preprocessing/include)

set(NAME MuonFeaturesExtraction)
set(TARGET_NAME ${NAME}.test)

add_executable(${TARGET_NAME}
    ../catch.cpp
    ${NAME}.test.cu)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${TARGET_NAME} PRIVATE Muon)

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})