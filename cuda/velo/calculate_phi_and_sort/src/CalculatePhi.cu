#include "VeloDefinitions.cuh"
#include "CalculatePhiAndSort.cuh"
#include "math_constants.h" // PI
#include "VeloTools.cuh"

/**
 * @brief Calculates a phi side
 */
template<class T>
__device__ void calculate_phi_side(
  float* shared_hit_phis,
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  uint* hit_permutations,
  const unsigned int starting_module,
  T calculate_hit_phi
) {
  for (unsigned int module=starting_module; module<VeloTracking::n_modules; module += 2) {
    const auto hit_start = module_hitStarts[module];
    const auto hit_num = module_hitNums[module];

    assert(hit_num < VeloTracking::max_numhits_in_module);

    // Calculate phis
    for (unsigned int i=0; i<(hit_num + blockDim.x - 1) / blockDim.x; ++i) {
      const auto hit_rel_id = i*blockDim.x + threadIdx.x;
      if (hit_rel_id < hit_num) {
        const auto hit_index = hit_start + hit_rel_id;
        const auto hit_phi = calculate_hit_phi(hit_Xs[hit_index], hit_Ys[hit_index]);
        shared_hit_phis[hit_rel_id] = hit_phi;
      }
    }

    // shared_hit_phis
    __syncthreads();

    // Find the permutations given the phis in shared_hit_phis
    for (unsigned int i=0; i<(hit_num + blockDim.x - 1) / blockDim.x; ++i) {
      const auto hit_rel_id = i*blockDim.x + threadIdx.x;
      if (hit_rel_id < hit_num) {
        const auto hit_index = hit_start + hit_rel_id;
        const auto phi = shared_hit_phis[hit_rel_id];
        
        // Find out local position
        unsigned int position = 0;
        for (unsigned int j=0; j<hit_num; ++j) {
          const auto other_phi = shared_hit_phis[j];
          // Stable sorting
          position += phi>other_phi || (phi==other_phi && hit_rel_id>j);
        }
        assert(position < VeloTracking::max_numhits_in_module);

        // Store it in hit permutations and in hit_Phis, already ordered
        const auto global_position = hit_start + position;
        hit_permutations[global_position] = hit_index;
        hit_Phis[global_position] = phi;
      }
    }

    // shared_hit_phis
    __syncthreads();
  }
}

/**
 * @brief Calculates phi for each hit
 */
__device__ void calculate_phi(
  const unsigned int* module_hitStarts,
  const unsigned int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  float* hit_Phis,
  uint* hit_permutations
) {
  __shared__ float shared_hit_phis [VeloTracking::max_numhits_in_module];

  // Odd modules
  calculate_phi_side(
    (float*) &shared_hit_phis[0],
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Phis,
    hit_permutations,
    1,
    [] (const float x, const float y) { return hit_phi_odd(x, y); }
  );

  // Even modules
  calculate_phi_side(
    (float*) &shared_hit_phis[0],
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Phis,
    hit_permutations,
    0,
    [] (const float x, const float y) { return hit_phi_even(x, y); }
  );
}
