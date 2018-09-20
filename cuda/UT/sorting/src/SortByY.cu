#include "SortByY.cuh"

__global__ void sort_by_y(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_offsets,
  uint* dev_hit_permutations,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  
  UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  // Two UTHits objects are created: one typecasts the base_pointer assuming
  // the data is unsorted, the other assuming the data is sorted.
  // This makes sorting more readable
  UTHits unsorted_ut_hits, sorted_ut_hits;
  unsorted_ut_hits.typecast_unsorted(dev_ut_hits, dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]);
  sorted_ut_hits.typecast_sorted(dev_ut_hits, dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]);

  // // Prints out all hits
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  //   for (int i=0; i<4; ++i) {
  //     printf("Layer %i hits:\n", i);
  //     for (int s=dev_unique_x_sector_layer_offsets[i]; s<dev_unique_x_sector_layer_offsets[i+1]; ++s) {
  //       printf(" Sector group %i, x %f:\n", s, dev_unique_sector_xs[s]);
  //       uint group_offset = ut_hit_offsets.sector_group_offset(s);
  //       uint n_hits_group = ut_hit_offsets.sector_group_number_of_hits(s);
  //       for (int j=0; j<n_hits_group; ++j) {
  //         const auto hit_index = group_offset + j;
  //         printf("  yBegin = %f, yEnd = %f, zAtYEq0 = %f, xAtYEq0 = %f, weight = %f, highThreshold = %u \n",
  //          unsorted_ut_hits.yBegin[hit_index],
  //          unsorted_ut_hits.yEnd[hit_index],
  //          unsorted_ut_hits.zAtYEq0[hit_index],
  //          unsorted_ut_hits.xAtYEq0[hit_index],
  //          unsorted_ut_hits.weight[hit_index],
  //          unsorted_ut_hits.highThreshold[hit_index]);
  //       }
  //     }
  //   }
  // }

  uint total_number_of_hits = 0;
  for (int sector_group=0; sector_group<number_of_unique_x_sectors; ++sector_group) {
    const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group);
    const uint sector_group_number_of_hits = ut_hit_offsets.sector_group_number_of_hits(sector_group);
    total_number_of_hits += sector_group_number_of_hits;

    find_permutation<float>(
      unsorted_ut_hits.xAtYEq0,
      sector_group_offset,
      dev_hit_permutations,
      sector_group_number_of_hits
    );
  }

  // A thread may have filled in a value in dev_hit_permutations and another
  // one may be using it in the next step
  __syncthreads();

  // Important note: Order matters, and should be kept as is
  apply_permutation<uint>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.planeCode, sorted_ut_hits.planeCode);
  __syncthreads();
  apply_permutation<uint>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.LHCbID, sorted_ut_hits.LHCbID);
  __syncthreads();
  apply_permutation<uint>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.highThreshold, sorted_ut_hits.highThreshold);
  __syncthreads();
  apply_permutation<float>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.weight, sorted_ut_hits.weight);
  __syncthreads();
  apply_permutation<float>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.xAtYEq0, sorted_ut_hits.xAtYEq0);
  __syncthreads();
  apply_permutation<float>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.zAtYEq0, sorted_ut_hits.zAtYEq0);
  __syncthreads();
  apply_permutation<float>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.yEnd, sorted_ut_hits.yEnd);
  __syncthreads();
  apply_permutation<float>(dev_hit_permutations, ut_hit_offsets.layer_offset(0), total_number_of_hits, unsorted_ut_hits.yBegin, sorted_ut_hits.yBegin);
}
