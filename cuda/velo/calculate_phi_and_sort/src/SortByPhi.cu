#include "VeloDefinitions.cuh"
#include "math_constants.h"

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__device__ void apply_permutation(
  uint* permutation,
  const uint event_hit_start,
  const uint event_number_of_hits,
  T* prev_container,
  T* new_container
) {
  // Apply permutation across all hits in the module (coalesced)
  for (uint i=0; i<(event_number_of_hits + blockDim.x - 1) / blockDim.x; ++i) {
    const auto permutation_index = i*blockDim.x + threadIdx.x;
    if (permutation_index < event_number_of_hits) {
      const auto hit_index = permutation[event_hit_start + permutation_index];
      new_container[event_hit_start + permutation_index] = prev_container[hit_index];
    }
  }
}

/**
 * @brief Calculates phi for each hit
 */
__device__ void sort_by_phi(
  const uint event_hit_start,
  const uint event_number_of_hits,
  float* hit_Xs,
  float* hit_Ys,
  float* hit_Zs,
  uint* hit_IDs,
  int32_t* hit_temp,
  uint* hit_permutations
) {
  // Let's work with new pointers
  // Note: It is important we populate later on in strictly
  //       the same order, to not lose data
  float* new_hit_Xs = (float*) hit_temp;
  float* new_hit_Ys = hit_Xs;
  float* new_hit_Zs = hit_Ys;
  uint* new_hit_IDs = (uint*) hit_Zs;
  
  // Apply permutation across all arrays
  apply_permutation(hit_permutations, event_hit_start, event_number_of_hits, hit_Xs, new_hit_Xs);
  __syncthreads();
  apply_permutation(hit_permutations, event_hit_start, event_number_of_hits, hit_Ys, new_hit_Ys);
  __syncthreads();
  apply_permutation(hit_permutations, event_hit_start, event_number_of_hits, hit_Zs, new_hit_Zs);
  __syncthreads();
  apply_permutation(hit_permutations, event_hit_start, event_number_of_hits, hit_IDs, new_hit_IDs);
}
