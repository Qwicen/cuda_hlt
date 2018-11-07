#include "VeloDefinitions.cuh"
#include "math_constants.h"
#include "ApplyPermutation.cuh"

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
