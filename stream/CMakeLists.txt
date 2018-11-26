include_directories(gear/include)
include_directories(handlers/include)
include_directories(memory_manager/include)
include_directories(scheduler/include)
include_directories(sequence/include)
include_directories(setup/include)
include_directories(checkers/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/configuration/sequences)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/calculate_phi_and_sort/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/consolidate_tracks/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/mask_clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/search_by_triplet/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/simplified_kalman_filter/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/IsMuon/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/patPV/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/velo/clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/pv/include)

file(GLOB stream_gear "gear/src/*cu")
file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_memory_manager "memory_manager/src/*cu")
file(GLOB stream_scheduler "scheduler/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")
file(GLOB stream_sequence_cpp "sequence/src/*cpp")
file(GLOB stream_setup "setup/src/*cu")
file(GLOB stream_visitors_velo "visitors/velo/src/*cu")
file(GLOB stream_visitors_pv "visitors/PV/src/*cu")
file(GLOB stream_visitors_UT "visitors/UT/src/*cu")
file(GLOB stream_visitors_SciFi "visitors/SciFi/src/*cu")
file(GLOB stream_visitors_Muon "visitors/Muon/src/*cu")

cuda_add_library(Stream STATIC
  ${stream_gear}
  ${stream_handlers}
  ${stream_memory_manager}
  ${stream_scheduler}
  ${stream_sequence}
  ${stream_sequence_cpp}
  ${stream_setup}
  ${stream_visitors_velo}
  ${stream_visitors_pv}
  ${stream_visitors_UT}
  ${stream_visitors_SciFi}
  ${stream_visitors_Muon}
)

target_link_libraries(Stream
  Common
  Utils
  Velo
  PV
  UT
  x86Forward
  SciFi
  x86VeloUT
  Muon)

if ( ROOT_FOUND )
  target_compile_definitions(Stream PUBLIC WITH_ROOT)
  message("-- Found ROOT, setting WITH_ROOT to true")
  target_link_libraries(Stream
    ${ROOT_LIBRARIES})
endif()
