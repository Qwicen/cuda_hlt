#include "SciFiSortByX.cuh"
#include "FindPermutation.cuh"
#include "ApplyPermutation.cuh"

using namespace SciFi;

__global__ void scifi_sort_by_x(
  uint* scifi_hits,
  uint32_t* scifi_hit_count,
  uint* scifi_hit_permutations
) {
  // Taken from UT sorting
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint* zone_offsets = scifi_hit_count + event_number * SciFi::Constants::n_zones;
  const uint* n_hits_zones = scifi_hit_count + number_of_events * SciFi::Constants::n_zones + 1 + event_number * SciFi::Constants::n_zones;

  // Two SciFiHits objects are created: one typecasts the base_pointer assuming
  // the data is unsorted, the other assuming the data is sorted.
  // This makes sorting more readable
  SciFiHits unsorted_scifi_hits, sorted_scifi_hits;
  unsorted_scifi_hits.typecast_unsorted(scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_zones]);
  sorted_scifi_hits.typecast_sorted(scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_zones]);

  uint total_number_of_hits = 0;
  for (int i_zone=0; i_zone < SciFi::Constants::n_zones; ++i_zone) {
    const uint zone_offset = zone_offsets[i_zone];
    const uint n_hits_zone = n_hits_zones[i_zone];
    total_number_of_hits += n_hits_zone;

    find_permutation(
      zone_offset,
      zone_offset,
      n_hits_zone,
    	scifi_hit_permutations,
      [&unsorted_scifi_hits] (const int a, const int b) {
        if (unsorted_scifi_hits.x0[a] > unsorted_scifi_hits.x0[b]) { return 1; }
        if (unsorted_scifi_hits.x0[a] == unsorted_scifi_hits.x0[b]) { return 0; }
        return -1;
      }
    );

    // Skip padding
    for(uint i = zone_offset + n_hits_zone; i < zone_offsets[i_zone + 1]; i++) {
      scifi_hit_permutations[i] = i;
      total_number_of_hits++;
    }
  }

  // A thread may have filled in a value in scifi_hit_permutations and another
  // one may be using it in the next step
  __syncthreads();

  // Important note: Order matters, and should be kept as is
  apply_permutation<uint>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.hitZone, sorted_scifi_hits.hitZone );
  __syncthreads();
  apply_permutation<uint>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.planeCode, sorted_scifi_hits.planeCode );
  __syncthreads();
  apply_permutation<uint>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.LHCbID, sorted_scifi_hits.LHCbID );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.yMax, sorted_scifi_hits.yMax );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.yMin, sorted_scifi_hits.yMin );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.dzdy, sorted_scifi_hits.dzdy );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.dxdy, sorted_scifi_hits.dxdy );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.w, sorted_scifi_hits.w );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.z0, sorted_scifi_hits.z0 );
  __syncthreads();
  apply_permutation<float>( scifi_hit_permutations, zone_offsets[0], total_number_of_hits, unsorted_scifi_hits.x0, sorted_scifi_hits.x0 );

}
