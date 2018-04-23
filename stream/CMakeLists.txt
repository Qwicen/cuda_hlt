
include_directories(handlers/include)
include_directories(sequence/include)

file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")

cuda_add_library( libStream STATIC
  ${stream_handlers}
  ${stream_sequence} 
  )

target_link_libraries(libStream libcuVelo)
