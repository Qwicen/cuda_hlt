#pragma once

#include <stdint.h>
#include "VeloEventModel.cuh"
#include "Common.h"
#include "Handler.cuh"

__device__ float velo_kalman_filter_step(
  const float z,
  const float zhit,
  const float xhit,
  const float whit,
  float& x,
  float& tx,
  float& covXX,
  float& covXTx,
  float& covTxTx
);

/**
 * @brief Fit the track with a Kalman filter,
 *        allowing for some scattering at every hit
 */
template<bool upstream>
__device__ void simplified_fit(
  const Velo::Track& track,
  const Velo::State& stateAtBeamLine,
  Velo::State* velo_state
) {
  // backward = state.z > track.hits[0].z;
  const bool backward = stateAtBeamLine.z > track.hits[0].z;
  const int direction = (backward ? 1 : -1) * (upstream ? 1 : -1);
  const float noise2PerLayer = 1e-8 + 7e-6 * (stateAtBeamLine.tx * stateAtBeamLine.tx + stateAtBeamLine.ty * stateAtBeamLine.ty);

  // assume the hits are sorted,
  // but don't assume anything on the direction of sorting
  int firsthit = 0;
  int lasthit = track.hitsNum - 1;
  int dhit = 1;
  if ((track.hits[lasthit].z - track.hits[firsthit].z) * direction < 0) {
    const int temp = firsthit;
    firsthit = lasthit;
    lasthit = temp;
    dhit = -1;
  }

  // We filter x and y simultaneously but take them uncorrelated.
  // filter first the first hit.
  Velo::State state;
  state.x = track.hits[firsthit].x;
  state.y = track.hits[firsthit].y;
  state.z = track.hits[firsthit].z;
  state.tx = stateAtBeamLine.tx;
  state.ty = stateAtBeamLine.ty;

  // Initialize the covariance matrix
  state.c00 = VeloTracking::param_w_inverted;
  state.c11 = VeloTracking::param_w_inverted;
  state.c20 = 0.f;
  state.c31 = 0.f;
  state.c22 = 1.f;
  state.c33 = 1.f;

  // add remaining hits
  state.chi2 = 0.0f;
  for (uint i=firsthit + dhit; i!=lasthit + dhit; i+=dhit) {
    const auto hit_x = track.hits[i].x;
    const auto hit_y = track.hits[i].y;
    const auto hit_z = track.hits[i].z;
    
    // add the noise
    state.c22 += noise2PerLayer;
    state.c33 += noise2PerLayer;

    // filter X and filter Y
    state.chi2 += velo_kalman_filter_step(state.z, hit_z, hit_x, VeloTracking::param_w, state.x, state.tx, state.c00, state.c20, state.c22);
    state.chi2 += velo_kalman_filter_step(state.z, hit_z, hit_y, VeloTracking::param_w, state.y, state.ty, state.c11, state.c31, state.c33);
    
    // update z (note done in the filter, since needed only once)
    state.z = hit_z;
  }

  // add the noise at the last hit
  state.c22 += noise2PerLayer;
  state.c33 += noise2PerLayer;

  // finally, store the state
  *velo_state = state;
}

__global__ void velo_fit(
  const uint32_t* dev_velo_cluster_container,
  const uint* dev_module_cluster_start,
  const int* dev_atomics_storage,
  const Velo::Track* dev_tracks,
  Velo::State* dev_velo_states
);

ALGORITHM(velo_fit, velo_fit_t)
