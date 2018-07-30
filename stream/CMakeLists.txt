
include_directories(gear/include)
include_directories(handlers/include)
include_directories(memory_manager/include)
include_directories(scheduler/include)
include_directories(sequence/include)
include_directories(sequence_setup/include)

file(GLOB stream_gear "gear/src/*cu")
file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_memory_manager "memory_manager/src/*cu")
file(GLOB stream_scheduler "scheduler/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")
file(GLOB stream_sequence_cpp "sequence/src/*cpp")
file(GLOB stream_sequence_setup "sequence_setup/src/*cu")

cuda_add_library(Stream STATIC
  ${stream_gear}
  ${stream_handlers}
  ${stream_memory_manager}
  ${stream_scheduler}
  ${stream_sequence}
  ${stream_sequence_cpp}
  ${stream_sequence_setup}
  )

if ( ROOT_FOUND )
  target_link_libraries(Stream
  Velo
  VeloUT
  #x86VeloUT
  ${ROOT_LIBRAIRES})
else()
  target_link_libraries(Stream
  Velo
  VeloUT)
  #x86VeloUT)
endif()


