#include "../include/ConsolidateTracks.cuh"


/**
 * @brief Calculates the parameters according to a root means square fit
 */
 __device__ Velo::State means_square_fit(
  Velo::Consolidated::Hits& consolidated_hits,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint* hit_IDs,
  const Velo::TrackHits& track
) {
  Velo::State state;

  // Fit parameters
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;
  
  // Iterate over hits
  for (unsigned short h=0; h<track.hitsNum; ++h) {
    const auto hit_index = track.hits[h];
    const auto x = hit_Xs[hit_index];
    const auto y = hit_Ys[hit_index];
    const auto z = hit_Zs[hit_index];
    
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

    state.z = -(state.x * state.tx + state.y * state.ty) / (state.tx * state.tx + state.ty * state.ty);

    state.backward = state.z > consolidated_hits.z[0];
  }

  {
    // Covariance
    const auto m00 = s0;
    const auto m11 = u0;
    const auto m20 = sz - state.z * s0;
    const auto m31 = uz - state.z * u0;
    const auto m22 = sz2 - 2 * state.z * sz + state.z * state.z * s0;
    const auto m33 = uz2 - 2 * state.z * uz + state.z * state.z * u0;
    const auto den20 = 1.0f / (m22 * m00 - m20 * m20);
    const auto den31 = 1.0f / (m33 * m11 - m31 * m31);

    state.c00 = m22 * den20;
    state.c20 = -m20 * den20;
    state.c22 = m00 * den20;
    state.c11 = m33 * den31;
    state.c31 = -m31 * den31;
    state.c33 = m11 * den31;
  }

  {
    //=========================================================================
    // Chi2 / degrees-of-freedom of straight-line fit
    //=========================================================================
    float ch = 0.0f;
    int nDoF = -4;
    for (uint h=0; h<track.hitsNum; ++h) {
      const auto z = consolidated_hits.z[h];

      const auto x = state.x + state.tx * z;
      const auto y = state.y + state.ty * z;

      const auto dx = x - consolidated_hits.x[h];
      const auto dy = y - consolidated_hits.y[h];
      
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

  state.x = state.x + state.tx * state.z;
  state.y = state.y + state.ty * state.z;

  return state;
}

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const Velo::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_velo_track_hits,
  uint* dev_velo_states
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const Velo::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;

  // Consolidated datatypes
  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  Velo::Consolidated::States velo_states {dev_velo_states, velo_tracks.total_number_of_tracks};

  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs = (uint32_t*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);

  for (uint i=threadIdx.x; i<number_of_tracks_event; i+=blockDim.x) {
    Velo::Consolidated::Hits consolidated_hits = velo_tracks.get_hits(dev_velo_track_hits, i);
    const Velo::TrackHits track = event_tracks[i];

    auto populate = [&track] (uint32_t* __restrict__ a, uint32_t* __restrict__ b) {
      for (int i=0; i<track.hitsNum; ++i) {
        const auto hit_index = track.hits[i];
        a[i] = b[hit_index];
      }
    };

    populate((uint32_t*) consolidated_hits.x, (uint32_t*) hit_Xs);
    populate((uint32_t*) consolidated_hits.y, (uint32_t*) hit_Ys);
    populate((uint32_t*) consolidated_hits.z, (uint32_t*) hit_Zs);
    populate((uint32_t*) consolidated_hits.LHCbID, (uint32_t*) hit_IDs);

    // Calculate and store fit in consolidated container
    Velo::State beam_state = means_square_fit(
      consolidated_hits,
      hit_Xs,
      hit_Ys,
      hit_Zs,
      hit_IDs,
      track
    );

    velo_states.set(event_tracks_offset + i, beam_state);
  }
}
