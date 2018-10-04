#include "SortByY.cuh"

__global__ void ut_apply_permutation(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_offsets,
  uint* dev_hit_permutations,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs,
  const uint number_of_events
) {
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];

  // Two UTHits objects are created: one typecasts the base_pointer assuming
  // the data is unsorted, the other assuming the data is sorted.
  // This makes sorting more readable
  UTHits unsorted_ut_hits, sorted_ut_hits;
  unsorted_ut_hits.typecast_unsorted(dev_ut_hits, total_number_of_hits);
  sorted_ut_hits.typecast_sorted(dev_ut_hits, total_number_of_hits);

  // Important note: Order matters, and should be kept as is
  const uint start_offset = blockIdx.x * blockDim.x;
  uint number_of_hits = blockDim.x;

  if ((start_offset + number_of_hits) > total_number_of_hits) {
    number_of_hits = total_number_of_hits - start_offset;
  }

  assert(start_offset < total_number_of_hits);
  assert((start_offset + number_of_hits) <= total_number_of_hits);

  apply_permutation<float>(dev_hit_permutations, start_offset, number_of_hits, unsorted_ut_hits.yBegin, sorted_ut_hits.yBegin);
  apply_permutation<uint32_t>(dev_hit_permutations, start_offset, number_of_hits, unsorted_ut_hits.raw_bank_index, sorted_ut_hits.raw_bank_index);
}

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

    find_permutation(
      sector_group_offset,
      dev_hit_permutations,
      sector_group_number_of_hits,
      [&unsorted_ut_hits] (const int a, const int b) {
        if (unsorted_ut_hits.yBegin[a] > unsorted_ut_hits.yBegin[b]) { return 1; }
        if (unsorted_ut_hits.yBegin[a] == unsorted_ut_hits.yBegin[b]) { return 0; }
        return -1;
      }
    );
  }
}
