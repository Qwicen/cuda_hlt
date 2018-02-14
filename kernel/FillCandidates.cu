#include "SearchByTriplet.cuh"

__device__ void fillCandidates(
  int* h0_candidates,
  int* h2_candidates,
  const int number_of_modules,
  const int* module_hitStarts,
  const int* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* module_Zs
) {
  // Notation is m0, m1, m2 in reverse order for each module
  // A hit in those is h0, h1, h2 respectively

  // Assign a module, h1 combination to each threadIdx.x
  int module_h1_counter = 0;
  for (int module_index=2; module_index<=49; ++module_index) {
    const auto m1_hitNums = module_hitNums[module_index];
    for (int i=0; i<(m1_hitNums + blockDim.x - 1) / blockDim.x; ++i) {
      const auto h1_rel_index = i*blockDim.x + threadIdx.x;

      if (h1_rel_index < m1_hitNums) {
        // Find for module module_index, hit h1_rel_index the candidates
        const auto m0_z = module_Zs[module_index+2];
        const auto m1_z = module_Zs[module_index];
        const auto m2_z = module_Zs[module_index-2];
        const auto m0_hitStarts = module_hitStarts[module_index+2];
        const auto m2_hitStarts = module_hitStarts[module_index-2];
        const auto m0_hitNums = module_hitNums[module_index+2];
        const auto m2_hitNums = module_hitNums[module_index-2];
        const auto h1_index = module_hitStarts[module_index] + h1_rel_index;
        const auto h1_x = hit_Xs[h1_index];

        // Calculate x limits in h0 and h2
        // Note: f0(z) = alpha*z
        //       f2(z) = (alpha+beta)*z
        const auto tolerance_m0 = PARAM_TOLERANCE_ALPHA * (m0_z - m1_z);
        const auto tolerance_m2 = (PARAM_TOLERANCE_ALPHA + PARAM_TOLERANCE_BETA) * (m1_z - m2_z);

        // Find candidates
        bool first_h0_found = false, last_h0_found = false;
        bool first_h2_found = false, last_h2_found = false;
        
        // Add h0 candidates
        for (int h0_index=m0_hitStarts; h0_index < m0_hitStarts + m0_hitNums; ++h0_index) {
          const auto h0_x = hit_Xs[h0_index];
          const bool tolerance_condition = fabs(h1_x - h0_x) < tolerance_m0;

          if (!first_h0_found && tolerance_condition) {
            h0_candidates[2*h1_index] = h0_index;
            first_h0_found = true;
          }
          else if (first_h0_found && !last_h0_found && !tolerance_condition) {
            h0_candidates[2*h1_index + 1] = h0_index;
            last_h0_found = true;
          }
        }
        if (first_h0_found && !last_h0_found) {
          h0_candidates[2*h1_index + 1] = m0_hitStarts + m0_hitNums;
        }

        // Add h2 candidates
        for (int h2_index=m2_hitStarts; h2_index < m2_hitStarts + m2_hitNums; ++h2_index) {
          const auto h2_x = hit_Xs[h2_index];
          const bool tolerance_condition = fabs(h1_x - h2_x) < tolerance_m2;

          if (!first_h2_found && tolerance_condition) {
            h2_candidates[2*h1_index] = h2_index;
            first_h2_found = true;
          }
          else if (first_h2_found && !last_h2_found && !tolerance_condition) {
            h2_candidates[2*h1_index + 1] = h2_index;
            last_h2_found = true;
          }
        }
        if (first_h2_found && !last_h2_found) {
          h2_candidates[2*h1_index + 1] = m2_hitStarts + m2_hitNums;
        }
      }
    }
  }
}
