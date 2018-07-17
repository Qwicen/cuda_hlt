#include "../include/ConsolidateTracks.cuh"

/**
 * @brief Calculates the parameters according to a root means square fit
 */
 __device__ VeloState means_square_fit(
  const VeloTracking::Hit<mc_check_enabled>* velo_track_hits,
  const VeloTracking::TrackHits& track
) {
  VeloState state;

  // Fit parameters
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;
  
  // Iterate over hits
  for (unsigned short h=0; h<track.hitsNum; ++h) {
    const auto hit = velo_track_hits[h];
    const auto x = hit.x;
    const auto y = hit.y;
    const auto z = hit.z;
    
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

    state.backward = state.z > velo_track_hits[0].z;
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
      const auto hit = velo_track_hits[h];
      const auto z = hit.z;

      const auto x = state.x + state.tx * z;
      const auto y = state.y + state.ty * z;

      const auto dx = x - hit.x;
      const auto dy = y - hit.y;
      
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
  const VeloTracking::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const VeloTracking::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  int* tracks_insert_pointer = dev_atomics_storage + event_number;
  const int number_of_tracks = *tracks_insert_pointer;

  // Store accumulated tracks after the number of tracks
  // Reusing the previous space for the weak tracks counter,
  // but storing in SoA rather than AoS now
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const auto accumulated_tracks = accumulated_tracks_base_pointer[event_number];
    
  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs = (uint32_t*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);

  // Consolidate tracks in dev_output_tracks
  VeloState* velo_states = dev_velo_states + 2* accumulated_tracks;

  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {
    const uint element = i * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      const VeloTracking::TrackHits track = event_tracks[element];

      // Pointer to velo_track_hit container
      VeloTracking::Hit<mc_check_enabled>* velo_track_hits = dev_velo_track_hits +
        dev_velo_track_hit_number[accumulated_tracks + element];

      // Store X, Y, Z and ID in consolidated container
      for (uint j=0; j<track.hitsNum; ++j) {
        const auto hit_index = track.hits[j];
        velo_track_hits[j] = VeloTracking::Hit<mc_check_enabled>{
          hit_Xs[hit_index],
          hit_Ys[hit_index],
          hit_Zs[hit_index]
#ifdef MC_CHECK
          ,hit_IDs[hit_index]
#endif
        };
      }

      // Calculate and store fit in consolidated container
      velo_states[2*element] = means_square_fit(
        velo_track_hits,
        track
      );

    }
  }
}
