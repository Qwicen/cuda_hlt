
include_directories(handlers/include)
include_directories(sequence/include)

find_package( ROOT REQUIRED ) 

file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")
file(GLOB stream_cpp "sequence/src/*cpp")

cuda_add_library(Stream STATIC
  ${stream_handlers}
  ${stream_sequence}
  ${stream_cpp}
  )

target_link_libraries(Stream Velo VeloUT ${ROOT_LIBRAIRES})
