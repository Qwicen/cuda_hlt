add_subdirectory(muon_catboost)
add_subdirectory(muon_features_extraction)

#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#include_directories(${CMAKE_SOURCE_DIR}/main/include)
#include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
#include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
#include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include)
#include_directories(${CMAKE_SOURCE_DIR}/stream/sequence/include)
#include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
#include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
#include_directories(${CMAKE_SOURCE_DIR}/mdf/include)
#
#set(UNIT_TEST_LIST
##  MuonFeaturesExtraction
#  MuonCatboost)
#
#set(CB_SRC
#  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/features_generated.h
#  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/ctr_data_generated.h
#  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/model_generated.h
#  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/src/evaluator.cpp)
#
#
#foreach(NAME IN LISTS UNIT_TEST_LIST)
#
#  set(TARGET_NAME ${NAME}.test)
#
#  add_executable(${TARGET_NAME}
#    main.cu
#    ${NAME}.test.cpp
#    ${CB_SRC}
#    MuonCatboostKernel.cu)
#
#  set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#  
#  target_link_libraries(${TARGET_NAME}
#    PRIVATE 
#    Muon
#    Stream
#    TrackChecking
#    PVChecking
#    x86Clustering
#    )
#  
#  add_test(
#    NAME ${TARGET_NAME}
#    COMMAND ${TARGET_NAME})
#endforeach()
