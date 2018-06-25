
include_directories(handlers/include)
include_directories(sequence/include)

find_package( ROOT REQUIRED ) 

file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")

cuda_add_library(Stream STATIC
  ${stream_handlers}
  ${stream_sequence} 
  )

target_link_libraries(Stream Velo VeloUT ${ROOT_LIBRAIRES})
