#include "SortByX.cuh"

__global__ void sort_by_x(
  uint32_t* dev_ut_hits,
  uint32_t* dev_ut_hit_count,
  uint* dev_hit_permutations
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint* layer_offsets = dev_ut_hit_count + event_number * VeloUTTracking::n_layers;
  const uint* n_hits_layers = dev_ut_hit_count + number_of_events * VeloUTTracking::n_layers + 1 + event_number * VeloUTTracking::n_layers;

  // Two UTHits objects are created: one typecasts the base_pointer assuming
  // the data is unsorted, the other assuming the data is sorted.
  // This makes sorting more readable
  UTHits unsorted_ut_hits, sorted_ut_hits;
  unsorted_ut_hits.typecast_unsorted(dev_ut_hits, dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers]);
  sorted_ut_hits.typecast_sorted(dev_ut_hits, dev_ut_hit_count[number_of_events * VeloUTTracking::n_layers]);

  uint total_number_of_hits = 0;
  for (int i_layer=0; i_layer<VeloUTTracking::n_layers; ++i_layer) {
    const uint layer_offset = layer_offsets[i_layer];
    const uint n_hits_layer = n_hits_layers[i_layer];
    total_number_of_hits += n_hits_layer;

    find_permutation<float>( 
      unsorted_ut_hits.xAtYEq0,
      layer_offset,
    	dev_hit_permutations,
    	n_hits_layer
    );
  }

  // A thread may have filled in a value in dev_hit_permutations and another
  // one may be using it in the next step
  __syncthreads();

  // Important note: Order matters, and should be kept as is
  apply_permutation<uint>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.planeCode, sorted_ut_hits.planeCode );
  __syncthreads();
  apply_permutation<uint>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.LHCbID, sorted_ut_hits.LHCbID );
  __syncthreads();
  apply_permutation<uint>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.highThreshold, sorted_ut_hits.highThreshold );
  __syncthreads();
  apply_permutation<float>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.weight, sorted_ut_hits.weight );
  __syncthreads();
  apply_permutation<float>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.xAtYEq0, sorted_ut_hits.xAtYEq0 );
  __syncthreads();
  apply_permutation<float>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.zAtYEq0, sorted_ut_hits.zAtYEq0 );
  __syncthreads();
  apply_permutation<float>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.yEnd, sorted_ut_hits.yEnd );
  __syncthreads();
  apply_permutation<float>( dev_hit_permutations, layer_offsets[0], total_number_of_hits, unsorted_ut_hits.yBegin, sorted_ut_hits.yBegin );
}
