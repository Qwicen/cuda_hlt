file(GLOB velo_common "common/src/*cu")
file(GLOB velo_prefix_sum "prefix_sum/src/*cu")
file(GLOB velo_clustering_sources "mask_clustering/src/*cu")
file(GLOB velo_phi_and_sort "calculate_phi_and_sort/src/*cu")
file(GLOB velo_search_by_triplet "search_by_triplet/src/*cu")
file(GLOB velo_simplified_kalman_filter "simplified_kalman_filter/src/*cu")
file(GLOB velo_consolidate_tracks "consolidate_tracks/src/*cu")

include_directories(common/include)
include_directories(prefix_sum/include) 
include_directories(mask_clustering/include)
include_directories(calculate_phi_and_sort/include)
include_directories(search_by_triplet/include)
include_directories(simplified_kalman_filter/include) 
include_directories(consolidate_tracks/include)
include_directories(../../main/include)
include_directories(../../stream/handlers/include)
include_directories(../event_model/velo/include)
include_directories(../event_model/common/include)
include_directories(../utils/sorting/include)
include_directories(../utils/prefix_sum/include)

cuda_add_library(Velo STATIC
  ${velo_common}
  ${velo_clustering_sources}
  ${velo_prefix_sum}
  ${velo_phi_and_sort}
  ${velo_search_by_triplet}
  ${velo_simplified_kalman_filter}
  ${velo_consolidate_tracks}
)
