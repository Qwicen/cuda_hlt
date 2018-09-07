#include "SortByX.cuh"

__global__ void sort_by_x(
  VeloUTTracking::HitsSoA* dev_ut_hits,
  VeloUTTracking::HitsSoA* dev_ut_hits_sorted,
  uint* dev_hit_permutations
) {

  const int number_of_events = gridDim.x;
  const int event_number = blockIdx.x;

  VeloUTTracking::HitsSoA* hits_layers        = dev_ut_hits + event_number;
  VeloUTTracking::HitsSoA* hits_layers_sorted = dev_ut_hits_sorted + event_number;

  uint* hit_permutations = dev_hit_permutations + event_number * VeloUTTracking::max_numhits_per_event;
  for ( int i_layer = 0; i_layer < VeloUTTracking::n_layers; ++i_layer ) {
    const uint n_hits = hits_layers->n_hits_layers[i_layer];
    const uint layer_offset = hits_layers->layer_offset[i_layer];
    for (unsigned int i=0; i<(n_hits + blockDim.x - 1) / blockDim.x; ++i) {
      const auto index = i*blockDim.x + threadIdx.x;
      if (index < n_hits) {
        hit_permutations[layer_offset + index] = 0;
      }
    }
    
    __syncthreads();
    
   
    if ( threadIdx.x == 0 ) {
      hits_layers_sorted->n_hits_layers[i_layer] = n_hits;
      hits_layers_sorted->layer_offset[i_layer] = layer_offset;
    }
    
    find_permutation<float>( 
        hits_layers->m_xAtYEq0,
        layer_offset,
      	hit_permutations,
      	n_hits
      );

    __syncthreads();

    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_cos, hits_layers_sorted->m_cos );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_weight, hits_layers_sorted->m_weight );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_xAtYEq0, hits_layers_sorted->m_xAtYEq0 );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_yBegin, hits_layers_sorted->m_yBegin );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_yEnd, hits_layers_sorted->m_yEnd );
      apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->m_zAtYEq0, hits_layers_sorted->m_zAtYEq0 );
      apply_permutation<unsigned int>( hit_permutations, layer_offset, n_hits, hits_layers->m_LHCbID, hits_layers_sorted->m_LHCbID );
      apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers->m_planeCode, hits_layers_sorted->m_planeCode );
      apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers->m_highThreshold, hits_layers_sorted->m_highThreshold );
  }
  
}
