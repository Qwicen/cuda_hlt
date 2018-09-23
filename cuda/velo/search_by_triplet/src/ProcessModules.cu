#include "ProcessModules.cuh"
#include "TrackSeeding.cuh"
#include "TrackForwarding.cuh"

/**
 * @brief Processes modules in decreasing order with some stride
 */
__device__ void process_modules(
  Velo::Module* module_data,
  float* shared_best_fits,
  const uint starting_module,
  const uint stride,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const uint number_of_modules,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const float* hit_Phis,
  uint* weaktracks_insert_pointer,
  uint* tracklets_insert_pointer,
  uint* ttf_insert_pointer,
  uint* tracks_insert_pointer,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits,
  unsigned short* h1_rel_indices,
  uint* local_number_of_hits,
  const uint hit_offset,
  const float* dev_velo_module_zs
) {
  auto first_module = starting_module;

  // Prepare the first seeding iteration
  // Load shared module information
  if (threadIdx.x < 6) {
    const auto module_number = first_module - threadIdx.x;
    module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
    module_data[threadIdx.x].hitNums = module_hitNums[module_number];
    module_data[threadIdx.x].z = dev_velo_module_zs[module_number];
  }

  // Due to shared module data loading
  __syncthreads();

  // Do first track seeding
  track_seeding(
    shared_best_fits,
    hit_Xs,
    hit_Ys,
    hit_Zs,
    module_data,
    h0_candidates,
    h2_candidates,
    hit_used,
    tracklets_insert_pointer,
    ttf_insert_pointer,
    tracklets,
    tracks_to_follow,
    h1_rel_indices,
    local_number_of_hits
  );

  // Prepare forwarding - seeding loop
  uint last_ttf = 0;
  first_module -= stride;

  while (first_module >= 4) {

    // Due to WAR between trackSeedingFirst and the code below
    __syncthreads();
    
    // Iterate in modules
    // Load in shared
    if (threadIdx.x < 6) {
      const auto module_number = first_module - threadIdx.x;
      module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
      module_data[threadIdx.x].hitNums = module_hitNums[module_number];
      module_data[threadIdx.x].z = dev_velo_module_zs[module_number];
    }

    const auto prev_ttf = last_ttf;
    last_ttf = ttf_insert_pointer[0];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset atomics
    local_number_of_hits[0] = 0;

    // Due to module data loading
    __syncthreads();

    // Track Forwarding
    track_forwarding(
      hit_Xs,
      hit_Ys,
      hit_Zs,
      hit_Phis,
      hit_used,
      tracks_insert_pointer,
      ttf_insert_pointer,
      weaktracks_insert_pointer,
      module_data,
      diff_ttf,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      number_of_hits
    );

    // Due to ttf_insert_pointer
    __syncthreads();

    // Seeding
    track_seeding(
      shared_best_fits,
      hit_Xs,
      hit_Ys,
      hit_Zs,
      module_data,
      h0_candidates,
      h2_candidates,
      hit_used,
      tracklets_insert_pointer,
      ttf_insert_pointer,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      local_number_of_hits
    );

    first_module -= stride;
  }

  // Due to last seeding ttf_insert_pointer
  __syncthreads();

  const auto prev_ttf = last_ttf;
  last_ttf = ttf_insert_pointer[0];
  const auto diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (int i=0; i<(diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const auto ttf_element = blockDim.x * i + threadIdx.x;

    if (ttf_element < diff_ttf) {
      const int fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % VeloTracking::ttf_modulo];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const auto weakP = atomicAdd(weaktracks_insert_pointer, 1);
        assert(weakP < number_of_hits);
        weak_tracks[weakP] = tracklets[trackno];
      }
    }
  }
}
