
include_directories(gear/include)
include_directories(handlers/include)
include_directories(memory_manager/include)
include_directories(scheduler/include)
include_directories(sequence/include)
include_directories(sequence_setup/include)
include_directories(../main/include)
include_directories(../cuda/utils/prefix_sum/include)
include_directories(../cuda/utils/sorting/include)
include_directories(../cuda/event_model/velo/include)
include_directories(../cuda/event_model/common/include)
include_directories(../cuda/UT/common/include)
include_directories(../cuda/UT/PrVeloUT/include)
include_directories(../cuda/UT/UTDecoding/include)
include_directories(../cuda/UT/sorting/include)
include_directories(../cuda/velo/common/include)
include_directories(../cuda/velo/calculate_phi_and_sort/include)
include_directories(../cuda/velo/consolidate_tracks/include)
include_directories(../cuda/velo/mask_clustering/include)
include_directories(../cuda/velo/prefix_sum/include)
include_directories(../cuda/velo/search_by_triplet/include)
include_directories(../cuda/velo/simplified_kalman_filter/include)
include_directories(../cuda/SciFi/preprocessing/include)
include_directories(../cuda/SciFi/common/include)
include_directories(../x86/velo/clustering/include)
include_directories(../checker/tracking/include)
include_directories(../x86/UT/PrVeloUT/include)
include_directories(../cuda/SciFi/common/include)
include_directories(../cuda/muon/common/include)
include_directories(../cuda/SciFi/PrForward/include)
include_directories(../cuda/utils/include)
include_directories(../cuda/veloUT/sorting/include)
include_directories(../x86/SciFi/include)

file(GLOB stream_gear "gear/src/*cu")
file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_memory_manager "memory_manager/src/*cu")
file(GLOB stream_scheduler "scheduler/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")
file(GLOB stream_sequence_cpp "sequence/src/*cpp")
file(GLOB stream_sequence_setup "sequence_setup/src/*cu")
file(GLOB stream_sequence_visitors_velo "sequence_visitors/velo/src/*cu")
file(GLOB stream_sequence_visitors_UT "sequence_visitors/UT/src/*cu")
file(GLOB stream_sequence_visitors_SciFi "sequence_visitors/SciFi/src/*cu")

cuda_add_library(Stream STATIC
  ${stream_gear}
  ${stream_handlers}
  ${stream_memory_manager}
  ${stream_scheduler}
  ${stream_sequence}
  ${stream_sequence_cpp}
  ${stream_sequence_setup}
  ${stream_sequence_visitors_velo}
  ${stream_sequence_visitors_UT}
  ${stream_sequence_visitors_SciFi}
  )

if ( ROOT_FOUND )
  target_compile_definitions(Stream PUBLIC WITH_ROOT)
  message("-- Found ROOT, setting WITH_ROOT to true")
  target_link_libraries(Stream
  Utils
  Velo
  UT
  x86Forward
  SciFi
  x86VeloUT
  ${ROOT_LIBRARIES})
else()
  target_link_libraries(Stream
  Utils
  Velo
  UT
  x86Forward
  SciFi
  x86VeloUT)
endif()
