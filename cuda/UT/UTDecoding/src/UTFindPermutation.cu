#include "UTFindPermutation.cuh"
#include "FindPermutation.cuh"
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
  
  const UTHitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  const UTHits ut_hits {dev_ut_hits, dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  // // Prints out all hits
  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
  //   printf(" Sector group %i, x %f:\n", sector_group_number, dev_unique_sector_xs[sector_group_number]);
  //   uint group_offset = ut_hit_offsets.sector_group_offset(sector_group_number);
  //   uint n_hits_group = ut_hit_offsets.sector_group_number_of_hits(sector_group_number);
  //   for (int j=0; j<n_hits_group; ++j) {
  //     const auto hit_index = group_offset + j;
  //     // printf("  yBegin = %f, yEnd = %f, zAtYEq0 = %f, xAtYEq0 = %f, weight = %f, highThreshold = %u \n",
  //     //  ut_hits.yBegin[hit_index],
  //     //  ut_hits.yEnd[hit_index],
  //     //  ut_hits.zAtYEq0[hit_index],
  //     //  ut_hits.xAtYEq0[hit_index],
  //     //  ut_hits.weight[hit_index],
  //     //  ut_hits.highThreshold[hit_index]);
  //     printf("  yBegin = %f\n", ut_hits.yBegin[hit_index]);
  //   }
  // }

  const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group_number);
  const uint sector_group_number_of_hits = ut_hit_offsets.sector_group_number_of_hits(sector_group_number);

  if (sector_group_number_of_hits > 0) {
    // Load yBegin into a shared memory container
    // TODO: Find a proper maximum and cover corner cases
    __shared__ float s_y_begin [UTDecoding::ut_max_hits_shared_sector_group];
    assert(sector_group_number_of_hits < UTDecoding::ut_max_hits_shared_sector_group);

    for (int i=threadIdx.x; i<sector_group_number_of_hits; i+=blockDim.x) {
      s_y_begin[i] = ut_hits.yBegin[sector_group_offset + i];
    }

    __syncthreads();

    // Sort according to the natural order in s_y_begin
    // Store the permutation found into dev_hit_permutations
    find_permutation(
      0,
      sector_group_offset,
      sector_group_number_of_hits,
      dev_hit_permutations,
      [] (const int a, const int b) -> int {
        return (s_y_begin[a] > s_y_begin[b]) - (s_y_begin[a] < s_y_begin[b]);
      }
    );
  }
}
