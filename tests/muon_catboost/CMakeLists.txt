include_directories(${CMAKE_SOURCE_DIR}/tests)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/standalone_evaluator)

set(NAME MuonCatboost)
set(TARGET_NAME ${NAME}.test)
set(CB_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/standalone_evaluator/features_generated.h
  ${CMAKE_CURRENT_SOURCE_DIR}/standalone_evaluator/ctr_data_generated.h
  ${CMAKE_CURRENT_SOURCE_DIR}/standalone_evaluator/model_generated.h
  ${CMAKE_CURRENT_SOURCE_DIR}/standalone_evaluator/evaluator.cpp)

add_executable(${TARGET_NAME}
    ../catch.cpp
    ${NAME}.test.cpp
    ${CB_SRC}
    MuonCatboostKernel.cu)

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${TARGET_NAME} PRIVATE 
    Muon Stream TrackChecking PVChecking x86Clustering)

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})