#include "UTFindPermutation.cuh"
#include <cstdio>

__global__ void ut_find_permutation(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_offsets,
  uint* dev_hit_permutations,
  const uint* dev_unique_x_sector_layer_offsets,
  const uint* dev_unique_x_sector_offsets,
  const float* dev_unique_sector_xs)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint sector_group_number = blockIdx.y;
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

  const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group_number);
  const uint sector_group_number_of_hits = ut_hit_offsets.sector_group_number_of_hits(sector_group_number);

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
