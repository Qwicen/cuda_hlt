#include "SearchByTriplet.cuh"
#include "WeakTracksAdder.cuh"

/**
 * @brief Calculates the parameters according to a root means square fit
 *        and returns the chi2.
 */
 __device__ float means_square_fit_chi2(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const TrackHits& track
) {
  VeloState state;

  // Fit parameters
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;
  
  // Iterate over hits
  for (unsigned short h=0; h<track.hitsNum; ++h) {
    const auto hitno = track.hits[h];
    const auto x = hit_Xs[hitno];
    const auto y = hit_Ys[hitno];
    const auto z = hit_Zs[hitno];
    
    const auto wx = VeloTracking::param_w;
    const auto wx_t_x = wx * x;
    const auto wx_t_z = wx * z;
    s0 += wx;
    sx += wx_t_x;
    sz += wx_t_z;
    sxz += wx_t_x * z;
    sz2 += wx_t_z * z;

    const auto wy = VeloTracking::param_w;
    const auto wy_t_y = wy * y;
    const auto wy_t_z = wy * z;
    u0 += wy;
    uy += wy_t_y;
    uz += wy_t_z;
    uyz += wy_t_y * z;
    uz2 += wy_t_z * z;
  }

  {
    // Calculate tx, ty and backward
    const auto dens = 1.0f / (sz2 * s0 - sz * sz);
    state.tx = (sxz * s0 - sx * sz) * dens;
    state.x = (sx * sz2 - sxz * sz) * dens;

    const auto denu = 1.0f / (uz2 * u0 - uz * uz);
    state.ty = (uyz * u0 - uy * uz) * denu;
    state.y = (uy * uz2 - uyz * uz) * denu;
  }

  {
    //=========================================================================
    // Chi2 / degrees-of-freedom of straight-line fit
    //=========================================================================
    float ch = 0.0f;
    int nDoF = -4;
    for (uint h=0; h<track.hitsNum; ++h) {
      const auto hitno = track.hits[h];

      const auto z = hit_Zs[hitno];
      const auto x = state.x + state.tx * z;
      const auto y = state.y + state.ty * z;

      const auto dx = x - hit_Xs[hitno];
      const auto dy = y - hit_Ys[hitno];
      
      ch += dx * dx * VeloTracking::param_w + dy * dy * VeloTracking::param_w;

      // Nice :)
      // TODO: We can get rid of the X and Y read here
      // float sum_w_xzi_2 = CL_VeloTracking::param_w * x; // for each hit
      // float sum_w_xi_2 = CL_VeloTracking::param_w * hit_Xs[hitno]; // for each hit
      // ch = (sum_w_xzi_2 - sum_w_xi_2) + (sum_w_yzi_2 - sum_w_yi_2);

      nDoF += 2;
    }
    state.chi2 = ch / nDoF; 
  }

  return state.chi2;
}

__device__ void weak_tracks_adder(
  uint* weaktracks_insert_pointer,
  uint* tracks_insert_pointer,
  TrackHits* weak_tracks,
  TrackHits* tracks,
  bool* hit_used,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs
) {
  // Compute the weak tracks
  const auto weaktracks_total = weaktracks_insert_pointer[0];
  for (int i=0; i<(weaktracks_total + blockDim.x - 1) / blockDim.x; ++i) {
    const auto weaktrack_no = blockDim.x * i + threadIdx.x;
    if (weaktrack_no < weaktracks_total) {
      const TrackHits& t = weak_tracks[weaktrack_no];
      const bool any_used = hit_used[t.hits[0]] || hit_used[t.hits[1]] || hit_used[t.hits[2]];
      const float chi2 = means_square_fit_chi2(
        hit_Xs,
        hit_Ys,
        hit_Zs,
        t
      );

      // Store them in the tracks bag
      if (!any_used && chi2 < VeloTracking::max_chi2) {
        const uint trackno = atomicAdd(tracks_insert_pointer, 1);
        assert(trackno < VeloTracking::max_tracks);
        tracks[trackno] = t;
      }
    }
  }
}
