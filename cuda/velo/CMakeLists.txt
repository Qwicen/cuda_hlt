
file(GLOB velo_common "common/src/*cu")
file(GLOB velo_clustering_sources "mask_clustering/src/*cu")
file(GLOB velo_phi_and_sort "calculate_phi_and_sort/src/*cu")
file(GLOB velo_search_by_triplet "search_by_triplet/src/*cu")
file(GLOB velo_simplified_kalman_filter "simplified_kalman_filter/src/*cu")
file(GLOB velo_consolidate_tracks "consolidate_tracks/src/*cu")

cuda_add_library( libcuVelo STATIC
  ${velo_common}
  ${velo_clustering_sources}
  ${velo_phi_and_sort}
  ${velo_search_by_triplet}
  ${velo_simplified_kalman_filter}
  ${velo_consolidate_tracks}
  )

# attention, the PUBLIC scope might cause problems if more projects
# are added which have headers with the same name
# should restructure project to not need this
# currently it is needed because some headers are included by files in stream
# we could pack everything cuda/velo related into the library
target_include_directories(libcuVelo PUBLIC common/include)
target_include_directories(libcuVelo PRIVATE mask_clustering/include)
target_include_directories(libcuVelo PRIVATE calculate_phi_and_sort/include)
target_include_directories(libcuVelo PRIVATE search_by_triplet/include)
target_include_directories(libcuVelo PRIVATE simplified_kalman_filter/include)
target_include_directories(libcuVelo PRIVATE consolidate_tracks/include)
