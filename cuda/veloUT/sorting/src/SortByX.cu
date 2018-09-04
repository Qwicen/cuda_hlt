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
        hits_layers->xAtYEq0,
        layer_offset,
      	hit_permutations,
      	n_hits
      );

    __syncthreads();

    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->weight, hits_layers_sorted->weight );
    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->xAtYEq0, hits_layers_sorted->xAtYEq0 );
    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->yBegin, hits_layers_sorted->yBegin );
    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->yEnd, hits_layers_sorted->yEnd );
    apply_permutation<float>( hit_permutations, layer_offset, n_hits, hits_layers->zAtYEq0, hits_layers_sorted->zAtYEq0 );
    apply_permutation<unsigned int>( hit_permutations, layer_offset, n_hits, hits_layers->LHCbID, hits_layers_sorted->LHCbID );
    apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers->planeCode, hits_layers_sorted->planeCode );
    apply_permutation<int>( hit_permutations, layer_offset, n_hits, hits_layers->highThreshold, hits_layers_sorted->highThreshold );
  }
  
}
