#include "ProcessModules.cuh"
#include "TrackSeedingFirst.cuh"
#include "TrackSeeding.cuh"
#include "TrackForwarding.cuh"

/**
 * @brief Processes modules in decreasing order with some stride
 */
__device__ void processModules(
  VeloTracking::Module* module_data,
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
  uint* weaktracks_insert_pointer,
  uint* tracklets_insert_pointer,
  uint* ttf_insert_pointer,
  uint* tracks_insert_pointer,
  uint* tracks_to_follow,
  uint* weak_tracks,
  VeloTracking::TrackHits* tracklets,
  VeloTracking::TrackHits* tracks,
  const uint number_of_hits,
  unsigned short* h1_rel_indices,
  uint* local_number_of_hits,
  const uint hit_offset
) {
  auto first_module = starting_module;

  // Prepare the first seeding iteration
  // Load shared module information
  if (threadIdx.x < 3) {
    const auto module_number = first_module - threadIdx.x * 2;
    module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
    module_data[threadIdx.x].hitNums = module_hitNums[module_number];
    module_data[threadIdx.x].z = VeloTracking::velo_module_zs[module_number];
  }

  // Due to shared module data loading
  __syncthreads();

  // Do first track seeding
  trackSeedingFirst(
    shared_best_fits,
    hit_Xs,
    hit_Ys,
    module_data,
    h0_candidates,
    h2_candidates,
    tracklets_insert_pointer,
    ttf_insert_pointer,
    tracklets,
    tracks_to_follow
  );

  // Prepare forwarding - seeding loop
  uint last_ttf = 0;
  uint last_weak_tracks = 0;
  first_module -= stride;

  while (first_module >= 4) {

    // Due to WAR between trackSeedingFirst and the code below
    __syncthreads();
    
    // Iterate in modules
    // Load in shared
    if (threadIdx.x < 3) {
      const int module_number = first_module - threadIdx.x * 2;
      module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
      module_data[threadIdx.x].hitNums = module_hitNums[module_number];
      module_data[threadIdx.x].z = VeloTracking::velo_module_zs[module_number];
    }

    const auto prev_ttf = last_ttf;
    last_ttf = ttf_insert_pointer[0];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset atomics
    local_number_of_hits[0] = 0;

    const auto prev_weak_tracks = last_weak_tracks;
    last_weak_tracks = weaktracks_insert_pointer[0];
    const auto diff_weak_tracks = last_weak_tracks - prev_weak_tracks;

    // Store weak tracks
    for (int i=0; i<(diff_weak_tracks + blockDim.x - 1) / blockDim.x; ++i) {
      const auto rel_weaktrack_no = blockDim.x * i + threadIdx.x;
      if (rel_weaktrack_no < diff_weak_tracks) {
        const auto weaktrack_no = (prev_weak_tracks + rel_weaktrack_no) % VeloTracking::ttf_modulo;;
        const VeloTracking::TrackHits t = tracklets[weak_tracks[weaktrack_no]];
        const bool any_used = hit_used[t.hits[0]] || hit_used[t.hits[1]] || hit_used[t.hits[2]];

        // Store them in the tracks bag
        if (!any_used) {
          const uint trackno = atomicAdd(tracks_insert_pointer, 1);
          assert(trackno < VeloTracking::max_tracks);
          tracks[trackno] = t;
        }
      }
    }

    // Due to module data loading
    __syncthreads();

    // Track Forwarding
    trackForwarding(
      hit_Xs,
      hit_Ys,
      hit_Zs,
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
    trackSeeding(
      shared_best_fits,
      hit_Xs,
      hit_Ys,
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
        // Store weak tracks if no hits are used
        const VeloTracking::TrackHits t = tracklets[trackno];
        const bool any_used = hit_used[t.hits[0]] || hit_used[t.hits[1]] || hit_used[t.hits[2]];

        // Store them in the tracks bag
        if (!any_used) {
          const uint trackno = atomicAdd(tracks_insert_pointer, 1);
          assert(trackno < VeloTracking::max_tracks);
          tracks[trackno] = t;
        }
      }
    }
  }
}
