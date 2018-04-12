#include "../include/CalculatePhiAndSort.cuh"

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void calculatePhiAndSort(
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint32_t* dev_velo_cluster_container,
  unsigned short* dev_hit_permutations
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;

  // Pointers to data within the event
  const uint cluster_offset = dev_module_cluster_start[52 * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * 52;
  const uint* module_hitNums = dev_module_cluster_num + event_number * 52;
  
  float* hit_Xs = (float*) (dev_velo_cluster_container + cluster_offset);
  float* hit_Ys = (float*) (hit_Xs + cluster_offset);
  float* hit_Zs = (float*) (hit_Ys + cluster_offset);
  uint32_t* hit_IDs = (uint32_t*) (hit_Zs + cluster_offset);
  float* hit_Phis = (float*) (hit_IDs + cluster_offset);
  int32_t* hit_temp = (int32_t*) (hit_Phis + cluster_offset);

  unsigned short* hit_permutations = dev_hit_permutations + cluster_offset;

  // Calculate phi and populate hit_permutations
  calculatePhi(
    module_hitStarts,
    module_hitNums,
    hit_Xs,
    hit_Ys,
    hit_Phis,
    hit_permutations
  );

  // Due to phi RAW
  __syncthreads();

  // Sort by phi
  sortByPhi(
    cluster_offset,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    hit_IDs,
    hit_temp,
    hit_permutations
  );
}
