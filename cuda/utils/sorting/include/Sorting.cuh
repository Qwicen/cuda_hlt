#pragma once

#include <cassert>

//--------------------------------------------------------
// 2018-08 Daniel Campora, Dorothea vom Bruch
//
//--------------------------------------------------------

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__host__ __device__ void apply_permutation(
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
void find_permutation(
  const uint hit_start,
  uint* hit_permutations,
  const uint n_hits,
  const T& sort_function
){
#ifdef __CUDA_ARCH__
  for (uint i=threadIdx.x; i<n_hits; i+=blockDim.x) {
    const int hit_index = hit_start + i;
    
    // Find out local position
    uint position = 0;
    for (uint j = 0; j < n_hits; ++j) {
      const int other_hit_index = hit_start + j;
      const int sort_result = sort_function(hit_index, other_hit_index);
      // Stable sort
      position += sort_result>0 || (sort_result==0 && i>j);
    }
    assert(position < n_hits);
    
    // Store it in hit_permutations 
    hit_permutations[hit_start + position] = hit_index; 
  }
#else
  for (uint i = 0; i < n_hits; ++i) {
    const int hit_index = hit_start + i;
    
    // Find out local position
    uint position = 0;
    for (uint j = 0; j < n_hits; ++j) {
      const int other_hit_index = hit_start + j;
      const int sort_result = sort_function(hit_index, other_hit_index);
      position += sort_result>0 || (sort_result==0 && i>j);
    }
    assert(position < n_hits);
    
    // Store it in hit_permutations 
    hit_permutations[hit_start + position] = hit_index; 
  }
#endif
  
}
