# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/velo/clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/global_event_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/compassUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/calculate_phi_and_sort/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/consolidate_tracks/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/mask_clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/search_by_triplet/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/simplified_kalman_filter/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/pv/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/sequence/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/kalman/ParKalman/include)
include_directories(${CMAKE_SOURCE_DIR}/mdf/include)

set(UNIT_TEST_LIST
#  MuonFeaturesExtraction
  MuonCatboost)

set(CB_SRC
  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/features_generated.h
  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/ctr_data_generated.h
  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include/model_generated.h
  ${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/src/evaluator.cpp)

file(GLOB common_sources "${CMAKE_SOURCE_DIR}/main/src/*")

# Remove main.cpp from common_sources
get_filename_component(main_cpp_path ${CMAKE_SOURCE_DIR}/main/src/main.cpp ABSOLUTE)
list(REMOVE_ITEM common_sources "${main_cpp_path}")

# common library
add_library(CommonLib ${common_sources})
target_link_libraries(CommonLib mdf)

set_property(TARGET CommonLib PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)

foreach(NAME IN LISTS UNIT_TEST_LIST)

  set(TARGET_NAME ${NAME}.test)

  add_executable(${TARGET_NAME}
    main.cu
    ${NAME}.test.cpp
    ${CB_SRC})

  set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  
  target_link_libraries(${TARGET_NAME}
    PRIVATE 
    Muon
    Stream
    TrackChecking
    PVChecking
    CheckClustering
    )
  
  add_test(
    NAME ${TARGET_NAME}
    COMMAND ${TARGET_NAME})
endforeach()
