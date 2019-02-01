include_directories(gear/include)
include_directories(handlers/include)
include_directories(memory_manager/include)
include_directories(scheduler/include)
include_directories(sequence/include)
include_directories(setup/include)
include_directories(checkers/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/float_operations/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/global_event_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/ip_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/compassUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/calculate_phi_and_sort/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/consolidate_tracks/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/mask_clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/search_by_triplet/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/simplified_kalman_filter/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/is_muon/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/muon/classification/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/patPV/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/velo/clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/pv/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/beamlinePV/include)
include_directories(${CMAKE_SOURCE_DIR}/x86/PV/beamlinePV/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/kalman/ParKalman/include)
include_directories(${PROJECT_BINARY_DIR}/configuration/sequences)

file(GLOB stream_gear "gear/src/*cu")
file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_memory_manager "memory_manager/src/*cu")
file(GLOB stream_scheduler "scheduler/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")
file(GLOB stream_sequence_cpp "sequence/src/*cpp")
file(GLOB stream_setup "setup/src/*cu")
file(GLOB stream_visitors_gec "visitors/global_event_cut/src/*cu")
file(GLOB stream_visitors_ip "visitors/ip_cut/src/*cu")
file(GLOB stream_visitors_velo "visitors/velo/src/*cu")
file(GLOB stream_visitors_patPV "visitors/patPV/src/*cu")
file(GLOB stream_visitors_beamlinePV "visitors/beamlinePV/src/*cu")
file(GLOB stream_visitors_assoc "visitors/associate/src/*cu")
file(GLOB stream_visitors_UT "visitors/UT/src/*cu")
file(GLOB stream_visitors_SciFi "visitors/SciFi/src/*cu")
file(GLOB stream_visitors_Muon "visitors/muon/src/*cu")
file(GLOB stream_visitors_kalman "visitors/kalman/src/*cu")

add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/configuration/sequences/ConfiguredSequence.h"
    COMMAND cmake -E copy "${CMAKE_SOURCE_DIR}/configuration/sequences/${SEQUENCE}.h" "${PROJECT_BINARY_DIR}/configuration/sequences/ConfiguredSequence.h"
    DEPENDS "${CMAKE_SOURCE_DIR}/configuration/sequences/${SEQUENCE}.h"
)
message(STATUS "Configured sequence: " ${SEQUENCE})

add_library(Stream STATIC
  "${PROJECT_BINARY_DIR}/configuration/sequences/ConfiguredSequence.h"
  ${stream_gear}
  ${stream_handlers}
  ${stream_memory_manager}
  ${stream_scheduler}
  ${stream_sequence}
  ${stream_sequence_cpp}
  ${stream_setup}
  ${stream_visitors_gec}
  ${stream_visitors_ip}
  ${stream_visitors_velo}
  ${stream_visitors_patPV}
  ${stream_visitors_beamlinePV}
  ${stream_visitors_assoc}
  ${stream_visitors_UT}
  ${stream_visitors_SciFi}
  ${stream_visitors_Muon}
  ${stream_visitors_kalman}
)

target_link_libraries(Stream PRIVATE
  Common
  Utils
  Associate
  IPCut
  GlobalEventCut
  Velo
  PatPV
  PV_beamline
  x86Clustering
  x86beamlinePV
  UT
  Kalman
  SciFi
  x86Forward
  x86VeloUT
  Muon)

set_property(TARGET Stream PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)

if (ROOT_FOUND)
  target_compile_definitions(Stream PUBLIC WITH_ROOT)
  target_include_directories(Stream BEFORE PRIVATE
    ${ROOT_INCLUDE_DIRS}
  )
  target_link_libraries(Stream PRIVATE
    ${ROOT_LIBRARIES}
  )
endif()
