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

  for (int i_layer=0; i_layer<VeloUTTracking::n_layers; ++i_layer) {
    const uint layer_offset = layer_offsets[i_layer];
    const uint n_hits_layer = n_hits_layers[i_layer];

    for (int j=threadIdx.x; j<n_hits_layers[i_layer]; j+=blockDim.x) {
      dev_hit_permutations[layer_offset + j] = 0;
    }

    __syncthreads();
    
    find_permutation<float>( 
        unsorted_ut_hits.m_xAtYEq0,
        layer_offset,
      	dev_hit_permutations,
      	n_hits_layer
      );

    __syncthreads();

    // Important note: Order matters, and should be kept as is
    apply_permutation<uint>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_planeCode, sorted_ut_hits.m_planeCode );
    apply_permutation<uint>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_LHCbID, sorted_ut_hits.m_LHCbID );
    apply_permutation<uint>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_highThreshold, sorted_ut_hits.m_highThreshold );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_weight, sorted_ut_hits.m_weight );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_xAtYEq0, sorted_ut_hits.m_xAtYEq0 );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_zAtYEq0, sorted_ut_hits.m_zAtYEq0 );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_yEnd, sorted_ut_hits.m_yEnd );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_yBegin, sorted_ut_hits.m_yBegin );
    apply_permutation<float>( dev_hit_permutations, layer_offset, n_hits_layer, unsorted_ut_hits.m_cos, sorted_ut_hits.m_cos );
  }
}
