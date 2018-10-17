#include "FillCandidates.cuh"
#include "VeloEventModel.cuh"
#include "VeloTools.cuh"
#include <cassert>
#include <cstdio>
#include <tuple>

__device__ void fill_candidates_impl(
  short* h0_candidates,
  short* h2_candidates,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Phis,
  const uint hit_offset
) {
  // Notation is m0, m1, m2 in reverse order for each module
  // A hit in those is h0, h1, h2 respectively

  // Assign a h1 to each threadIdx.x
  const auto module_index = blockIdx.y + 2; // 48 blocks y
  const auto m1_hitNums = module_hitNums[module_index];
  for (auto h1_rel_index=threadIdx.x; h1_rel_index<m1_hitNums; h1_rel_index+=blockDim.x) {
    // Find for module module_index, hit h1_rel_index the candidates
    const auto m0_hitStarts = module_hitStarts[module_index+2] - hit_offset;
    const auto m2_hitStarts = module_hitStarts[module_index-2] - hit_offset;
    const auto m0_hitNums = module_hitNums[module_index+2];
    const auto m2_hitNums = module_hitNums[module_index-2];

    const auto h1_index = module_hitStarts[module_index] + h1_rel_index - hit_offset;

    // Calculate phi limits
    const float h1_phi = hit_Phis[h1_index];

    int first_h0_bin = -1, last_h0_bin = -1;
    if (m0_hitNums > 0) {
      // Do a binary search for h0 candidates
      first_h0_bin = binary_search_first_candidate(
        hit_Phis + m0_hitStarts,
        m0_hitNums,
        h1_phi,
        VeloTracking::phi_extrapolation
      );

      if (first_h0_bin != -1) {
        // Find last h0 candidate
        last_h0_bin = binary_search_second_candidate(
          hit_Phis + m0_hitStarts + first_h0_bin,
          m0_hitNums - first_h0_bin,
          h1_phi,
          VeloTracking::phi_extrapolation
        );
        first_h0_bin += m0_hitStarts;
        last_h0_bin = last_h0_bin==0 ? first_h0_bin+1 : first_h0_bin+last_h0_bin;
      }
    }

    h0_candidates[2*h1_index] = first_h0_bin;
    h0_candidates[2*h1_index + 1] = last_h0_bin;

    int first_h2_bin = -1, last_h2_bin = -1;
    if (m2_hitNums > 0) {
      // Do a binary search for h2 candidates
      first_h2_bin = binary_search_first_candidate(
        hit_Phis + m2_hitStarts,
        m2_hitNums,
        h1_phi,
        VeloTracking::phi_extrapolation
      );

      if (first_h2_bin != -1) {
        // Find last h0 candidate
        last_h2_bin = binary_search_second_candidate(
          hit_Phis + m2_hitStarts + first_h2_bin,
          m2_hitNums - first_h2_bin,
          h1_phi,
          VeloTracking::phi_extrapolation
        );
        first_h2_bin += m2_hitStarts;
        last_h2_bin = last_h2_bin==0 ? first_h2_bin+1 : first_h2_bin+last_h2_bin;
      }
    }

    h2_candidates[2*h1_index] = first_h2_bin;
    h2_candidates[2*h1_index + 1] = last_h2_bin;
  }
}

__global__ void fill_candidates(
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  short* dev_h0_candidates,
  short* dev_h2_candidates
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;

  // Pointers to data within the event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint* module_hitNums = dev_module_cluster_num + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  assert((module_hitStarts[52] - module_hitStarts[0]) < VeloTracking::max_number_of_hits_per_event);
  
  // Order has changed since SortByPhi
  const float* hit_Phis = (float*) (dev_velo_cluster_container + 4 * number_of_hits + hit_offset);
  short* h0_candidates = dev_h0_candidates + 2*hit_offset;
  short* h2_candidates = dev_h2_candidates + 2*hit_offset;

  fill_candidates_impl(
    h0_candidates,
    h2_candidates,
    module_hitStarts,
    module_hitNums,
    hit_Phis,
    hit_offset
  );
}
