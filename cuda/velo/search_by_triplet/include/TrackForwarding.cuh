#pragma once

#include "VeloEventModel.cuh"
#include <tuple>

__device__ void track_forwarding(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const float* hit_Phis,
  bool* hit_used,
  uint* tracks_insertPointer,
  uint* ttf_insertPointer,
  uint* weaktracks_insertPointer,
  const Velo::Module* module_data,
  const uint diff_ttf,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  const uint prev_ttf,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits
);

/**
 * @brief Finds candidates in the specified module.
 */
template<typename T>
__device__ std::tuple<int, int> find_forward_candidates(
  const Velo::Module& module,
  const float tx,
  const float ty,
  const float* hit_Phis,
  const Velo::HitBase& h0,
  const T calculate_hit_phi
) {
  const auto dz = module.z - h0.z;
  const auto predx = tx * dz;
  const auto predy = ty * dz;
  const auto x_prediction = h0.x + predx;
  const auto y_prediction = h0.y + predy;
  const auto track_extrapolation_phi = calculate_hit_phi(x_prediction, y_prediction);

  int first_candidate = -1, last_candidate = -1;
  first_candidate = binary_search_first_candidate(
    hit_Phis + module.hitStart,
    module.hitNums,
    track_extrapolation_phi,
    VeloTracking::forward_phi_tolerance
  );

  if (first_candidate != -1) {
    // Find last candidate
    last_candidate = binary_search_second_candidate(
      hit_Phis + module.hitStart + first_candidate,
      module.hitNums - first_candidate,
      track_extrapolation_phi,
      VeloTracking::forward_phi_tolerance
    );
    first_candidate += module.hitStart;
    last_candidate = last_candidate==0 ? first_candidate+1 : first_candidate+last_candidate;
  }

  return {first_candidate, last_candidate};
}
