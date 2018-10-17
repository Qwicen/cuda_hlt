#include "SearchByTriplet.cuh"
#include "VeloTools.cuh"
#include <cstdio>

/**
 * @brief Fits hits to tracks.
 * 
 * @details In case the tolerances constraints are met,
 *          returns the chi2 weight of the track. Otherwise,
 *          returns FLT_MAX.
 */
__device__ float fit_hit_to_track(
  const Velo::HitBase& h0,
  const Velo::HitBase& h2,
  const float predx,
  const float predy,
  const float scatterDenom2
) {
  // tolerances
  const float x_prediction = h0.x + predx;
  const float dx = fabs(x_prediction - h2.x);
  const bool tolx_condition = dx < VeloTracking::tolerance;

  const float y_prediction = h0.y + predy;
  const float dy = fabs(y_prediction - h2.y);
  const bool toly_condition = dy < VeloTracking::tolerance;

  // Scatter
  const float scatterNum = (dx * dx) + (dy * dy);
  const float scatter = scatterNum * scatterDenom2;

  const bool scatter_condition = scatter < VeloTracking::max_scatter_forwarding;
  const bool condition = tolx_condition && toly_condition && scatter_condition;

  return condition * scatter + !condition * FLT_MAX;
}

/**
 * @brief Performs the track forwarding of forming tracks
 */
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
) {
  // Assign a track to follow to each thread
  for (int i=0; i<(diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const uint ttf_element = blockDim.x * i + threadIdx.x;
    if (ttf_element < diff_ttf) {
      const auto fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % VeloTracking::ttf_modulo];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const auto skipped_modules = (fulltrackno & 0x70000000) >> 28;
      auto trackno = fulltrackno & 0x0FFFFFFF;
      assert(track_flag ? trackno < VeloTracking::ttf_modulo : trackno < VeloTracking::max_tracks);

      Velo::TrackHits t = track_flag ? Velo::TrackHits{tracklets[trackno]} : tracks[trackno];

      // Load last two hits in h0, h1
      assert(t.hitsNum < VeloTracking::max_track_size);
      const auto h0_num = t.hits[t.hitsNum - 2];
      const auto h1_num = t.hits[t.hitsNum - 1];

      assert(h0_num < number_of_hits);
      const Velo::HitBase h0 {hit_Xs[h0_num], hit_Ys[h0_num], hit_Zs[h0_num]};

      assert(h1_num < number_of_hits);
      const Velo::HitBase h1 {hit_Xs[h1_num], hit_Ys[h1_num], hit_Zs[h1_num]};

      // Track forwarding over t, for all hits in the next module
      // Line calculations
      const auto td = 1.0f / (h1.z - h0.z);
      const auto txn = (h1.x - h0.x);
      const auto tyn = (h1.y - h0.y);
      const auto tx = txn * td;
      const auto ty = tyn * td;
      
      // Find the best candidate
      float best_fit = FLT_MAX;
      unsigned short best_h2;

      // Get candidates by performing a binary search in expected phi
      const auto odd_module_candidates = find_forward_candidates(
        module_data[4],
        tx,
        ty,
        hit_Phis,
        h0,
        [] (const float x, const float y) { return hit_phi_odd(x, y); }
      );

      const auto even_module_candidates = find_forward_candidates(
        module_data[5],
        tx,
        ty,
        hit_Phis,
        h0,
        [] (const float x, const float y) { return hit_phi_even(x, y); }
      );
      
      // Search on both modules in the same for loop
      const int total_odd_candidates = std::get<1>(odd_module_candidates) - std::get<0>(odd_module_candidates);
      const int total_even_candidates = std::get<1>(even_module_candidates) - std::get<0>(even_module_candidates);
      const int total_candidates = total_odd_candidates + total_even_candidates;

      for (int j=0; j<total_candidates; ++j) {
        const int h2_index = j < total_odd_candidates ?
          std::get<0>(odd_module_candidates) + j :
          std::get<0>(even_module_candidates) + j - total_odd_candidates;

        const Velo::HitBase h2 {hit_Xs[h2_index], hit_Ys[h2_index], hit_Zs[h2_index]};

        const auto dz = h2.z - h0.z;
        const auto predx = tx * dz;
        const auto predy = ty * dz;
        const auto scatterDenom2 = 1.f / ((h2.z - h1.z) * (h2.z - h1.z));

        const auto fit = fit_hit_to_track(
          h0,
          h2,
          predx,
          predy,
          scatterDenom2
        );
        
        // We keep the best one found
        if (fit < best_fit) {
          best_fit = fit;
          best_h2 = h2_index;
        }
      }

      // Condition for finding a h2
      if (best_fit != FLT_MAX) {
        // Mark h2 as used
        assert(best_h2 < number_of_hits);
        hit_used[best_h2] = true;

        // Update the tracks to follow, we'll have to follow up
        // this track on the next iteration :)
        assert(t.hitsNum < VeloTracking::max_track_size);
        t.hits[t.hitsNum++] = best_h2;

        // Update the track in the bag
        if (t.hitsNum <= 4) {
          assert(t.hits[0] < number_of_hits);
          assert(t.hits[1] < number_of_hits);
          assert(t.hits[2] < number_of_hits);

          // Also mark the first three as used
          hit_used[t.hits[0]] = true;
          hit_used[t.hits[1]] = true;
          hit_used[t.hits[2]] = true;

          // If it is a track made out of less than or equal than 4 hits,
          // we have to allocate it in the tracks pointer
          trackno = atomicAdd(tracks_insertPointer, 1);
        }

        // Copy the track into tracks
        assert(trackno < VeloTracking::max_tracks);
        tracks[trackno] = t;

        // Add the tracks to the bag of tracks to_follow
        const auto ttfP = atomicAdd(ttf_insertPointer, 1) % VeloTracking::ttf_modulo;
        tracks_to_follow[ttfP] = trackno;
      }
      // A track just skipped a module
      // We keep it for another round
      else if (skipped_modules < VeloTracking::max_skipped_modules) {
        // Form the new mask
        trackno = ((skipped_modules + 1) << 28) | (fulltrackno & 0x8FFFFFFF);

        // Add the tracks to the bag of tracks to_follow
        const auto ttfP = atomicAdd(ttf_insertPointer, 1) % VeloTracking::ttf_modulo;
        tracks_to_follow[ttfP] = trackno;
      }
      // If there are only three hits in this track,
      // mark it as "doubtful"
      else if (t.hitsNum == 3) {
        const auto weakP = atomicAdd(weaktracks_insertPointer, 1) % VeloTracking::ttf_modulo;
        assert(weakP < VeloTracking::max_weak_tracks);
        weak_tracks[weakP] = Velo::TrackletHits{t.hits[0], t.hits[1], t.hits[2]};
      }
      // In the "else" case, we couldn't follow up the track,
      // so we won't be track following it anymore.
    }
  }
}
