#pragma once

//--------------------------------------------------------
// 2018-08 Daniel Campora, Dorothea vom Bruch
//
//--------------------------------------------------------

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__host__ __device__ void applyPermutation(
  uint* permutation,
  const uint hit_start,
  const uint number_of_hits,
  T* prev_container,
  T* new_container
) {
  // Apply permutation across all hits
#ifdef __CUDA_ARCH__
  for (uint i=0; i<(number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto permutation_index = i*blockDim.x + threadIdx.x;
    if (permutation_index < number_of_hits) {
      const auto hit_index_global = permutation[hit_start + permutation_index];
      new_container[hit_start + permutation_index] = prev_container[hit_index_global];
    }
  }
#else
  for (uint i=0; i<number_of_hits; ++i) {
    const auto hit_index_global = permutation[hit_start + i];
    new_container[hit_start + i] = prev_container[hit_index_global];
  }
#endif 
}



/**
 * @brief Sort by var stored in sorting_vars, store index in hit_permutations
 */
template<class T>
__host__ __device__
void findPermutation(
  const T* sorting_vars,
  const uint hit_start,
  uint* hit_permutations,
  const uint n_hits
){
#ifdef __CUDA_ARCH__
  for (unsigned int i = 0; i < (n_hits + blockDim.x - 1); ++i) {
    const unsigned int hit_rel_index = i*blockDim.x + threadIdx.x;
    if ( hit_rel_index < n_hits ) {
      const int hit_index = hit_start + hit_rel_index;
      const T var = sorting_vars[hit_index];
      
      // Find out local position
      unsigned int position = 0;
      for (unsigned int j = 0; j < n_hits; ++j) {
        const int other_hit_index = hit_start + j;
        const T other_var = sorting_vars[other_hit_index];
        // Stable sorting
        position += var > other_var || ( var == other_var && hit_rel_index > j );
      }
      assert(position < n_hits);
      
      // Store it in hit_permutations 
      hit_permutations[hit_start + position] = hit_index; 
    }
  }
#else
  for (unsigned int i = 0; i < n_hits; ++i) {
    const int hit_index = hit_start + i;
    const T var = sorting_vars[hit_index];
    
    // Find out local position
    unsigned int position = 0;
    for (unsigned int j = 0; j < n_hits; ++j) {
      const int other_hit_index = hit_start + j;
      const T other_var = sorting_vars[other_hit_index];
      // Stable sorting
      position += var > other_var || ( var == other_var && i > j );
    }
    assert(position < n_hits);
    
    // Store it in hit_permutations 
    hit_permutations[hit_start + position] = hit_index; 
  }
#endif
  
}
