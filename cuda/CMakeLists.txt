cuda_add_library( libcuSearchByTriplet STATIC
  src/TrackForwarding.cu
  src/TrackSeeding.cu
  src/FillCandidates.cu
  src/SearchByTriplet.cu
  src/ProcessModules.cu
  src/TrackSeedingFirst.cu
  src/WeakTracksAdder.cu
  src/CalculatePhi.cu
  src/SortByPhi.cu
  src/CalculatePhiAndSort.cu
  src/VeloKalmanFilter.cu
  src/ConsolidateTracks.cu
  )

