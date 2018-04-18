cuda_add_library( libStream STATIC
  src/Stream.cu
  src/HandleCalculatePhiAndSort.cu
  src/HandleCalculateVeloStates.cu
  src/HandleSearchByTriplet.cu
  src/HandleConsolidateTracks.cu
  )

target_link_libraries( libStream libcuSearchByTriplet )
