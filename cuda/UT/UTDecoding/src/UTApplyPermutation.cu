#include "UTApplyPermutation.cuh"

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

  apply_permutation<uint32_t>(dev_hit_permutations, start_offset, number_of_hits, unsorted_ut_hits.raw_bank_index, sorted_ut_hits.raw_bank_index);
}
