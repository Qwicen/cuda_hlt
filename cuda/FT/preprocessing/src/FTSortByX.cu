#include "FTSortByX.cuh"

using namespace FT;

__global__ void ft_sort_by_x(
  char* dev_ft_hits,
  uint32_t* dev_ft_hit_count,
  uint* dev_ft_hit_permutations
) {
  // Taken from UT sorting
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint* zone_offsets = dev_ft_hit_count + event_number * FT::number_of_zones;
  const uint* n_hits_zones = dev_ft_hit_count + number_of_events * FT::number_of_zones + 1 + event_number * FT::number_of_zones;

  // Two FTHits objects are created: one typecasts the base_pointer assuming
  // the data is unsorted, the other assuming the data is sorted.
  // This makes sorting more readable
  FTHits unsorted_ft_hits, sorted_ft_hits;
  unsorted_ft_hits.typecast_unsorted(dev_ft_hits, dev_ft_hit_count[number_of_events * FT::number_of_zones]);
  sorted_ft_hits.typecast_sorted(dev_ft_hits, dev_ft_hit_count[number_of_events * FT::number_of_zones]);

  uint total_number_of_hits = 0;
  for (int i_zone=0; i_zone < FT::number_of_zones; ++i_zone) {
    const uint zone_offset = zone_offsets[i_zone];
    const uint n_hits_zone = n_hits_zones[i_zone];
    total_number_of_hits += n_hits_zone;

    find_permutation<float>(
      unsorted_ft_hits.x0,
      zone_offset,
    	dev_ft_hit_permutations,
    	n_hits_zone
    );

    // Skip padding
    for(uint i = zone_offset + n_hits_zone; i < zone_offsets[i_zone + 1]; i++) {
      dev_ft_hit_permutations[i] = i;
      total_number_of_hits++;
    }
  }

  // A thread may have filled in a value in dev_ft_hit_permutations and another
  // one may be using it in the next step
  __syncthreads();

  // Important note: Order matters, and should be kept as is
  apply_permutation<bool>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.used, sorted_ft_hits.used );
  __syncthreads();
  apply_permutation<uint>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.info, sorted_ft_hits.info );
  __syncthreads();
  apply_permutation<uint>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.hitZone, sorted_ft_hits.hitZone );
  __syncthreads();
  apply_permutation<uint>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.planeCode, sorted_ft_hits.planeCode );
  __syncthreads();
  apply_permutation<uint>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.LHCbID, sorted_ft_hits.LHCbID );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.coord, sorted_ft_hits.coord );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.werrX, sorted_ft_hits.werrX );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.yMax, sorted_ft_hits.yMax );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.yMin, sorted_ft_hits.yMin );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.dzdy, sorted_ft_hits.dzdy );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.dxdy, sorted_ft_hits.dxdy );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.w, sorted_ft_hits.w );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.z0, sorted_ft_hits.z0 );
  __syncthreads();
  apply_permutation<float>( dev_ft_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_ft_hits.x0, sorted_ft_hits.x0 );
}
