#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include <stdint.h>

/**
 * @brief Calculates the parameters according to a root means square fit
 */
__device__ void b_means_square_fit(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const TrackHits& track,
  TrackFitParameters& parameters,
  VeloState& state
) {
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

    state.z = -(state.x * state.tx + state.y * state.ty) / (state.tx * state.tx + state.ty * state.ty);
    parameters.backward = state.z > hit_Zs[track.hits[0]];
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

  state.x = state.x + state.tx * state.z;
  state.y = state.y + state.ty * state.z;

  // Keep some parameters around for the upcoming Kalman fit
  parameters.tx = state.tx;
  parameters.ty = state.ty;
}

template<bool mc_check_enabled>
__device__ Track<mc_check_enabled> createTrack(
  const TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
);

template<bool mc_check_enabled>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  Track <mc_check_enabled> * dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num
);

#include "ConsolidateTracks_impl.cuh"
