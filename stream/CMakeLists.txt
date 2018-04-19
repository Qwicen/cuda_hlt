
file(GLOB stream_handlers "handlers/src/*cu")
file(GLOB stream_sequence "sequence/src/*cu")

cuda_add_library( libStream STATIC
  ${stream_handlers}
  ${stream_sequence} 
  )

target_include_directories(libStream PRIVATE handlers/include)
target_include_directories(libStream PRIVATE sequence/include)

target_link_libraries(libStream libcuVelo)
