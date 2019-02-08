#pragma once

#include <stdint.h>
#include "VeloEventModel.cuh"
#include "States.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "VeloConsolidated.cuh"
#include "ArgumentsVelo.cuh"

__device__ float velo_kalman_filter_step(
  const float z,
  const float zhit,
  const float xhit,
  const float whit,
  float& x,
  float& tx,
  float& covXX,
  float& covXTx,
  float& covTxTx);

/**
 * @brief Fit the track with a Kalman filter,
 *        allowing for some scattering at every hit
 */
template<bool upstream>
__device__ VeloState
simplified_fit(const Velo::Consolidated::Hits consolidated_hits, const VeloState& stateAtBeamLine, const uint nhits)
{
  // backward = state.z > track.hits[0].z;
  const bool backward = stateAtBeamLine.z > consolidated_hits.z[0];
  const int direction = (backward ? 1 : -1) * (upstream ? 1 : -1);
  const float noise2PerLayer =
    1e-8 + 7e-6 * (stateAtBeamLine.tx * stateAtBeamLine.tx + stateAtBeamLine.ty * stateAtBeamLine.ty);

  // assume the hits are sorted,
  // but don't assume anything on the direction of sorting
  int firsthit = 0;
  int lasthit = nhits - 1;
  int dhit = 1;
  if ((consolidated_hits.z[lasthit] - consolidated_hits.z[firsthit]) * direction < 0) {
    const int temp = firsthit;
    firsthit = lasthit;
    lasthit = temp;
    dhit = -1;
  }

  // We filter x and y simultaneously but take them uncorrelated.
  // filter first the first hit.
  VeloState state;
  state.x = consolidated_hits.x[firsthit];
  state.y = consolidated_hits.y[firsthit];
  state.z = consolidated_hits.z[firsthit];
  state.tx = stateAtBeamLine.tx;
  state.ty = stateAtBeamLine.ty;

  // Initialize the covariance matrix
  state.c00 = Velo::Tracking::param_w_inverted;
  state.c11 = Velo::Tracking::param_w_inverted;
  state.c20 = 0.f;
  state.c31 = 0.f;
  state.c22 = 1.f;
  state.c33 = 1.f;

  // add remaining hits
  state.chi2 = 0.0f;
  for (uint i = firsthit + dhit; i != lasthit + dhit; i += dhit) {
    int hitindex = i;
    const auto hit_x = consolidated_hits.x[hitindex];
    const auto hit_y = consolidated_hits.y[hitindex];
    const auto hit_z = consolidated_hits.z[hitindex];

    // add the noise
    state.c22 += noise2PerLayer;
    state.c33 += noise2PerLayer;

    // filter X and filter Y
    state.chi2 += velo_kalman_filter_step(
      state.z, hit_z, hit_x, Velo::Tracking::param_w, state.x, state.tx, state.c00, state.c20, state.c22);
    state.chi2 += velo_kalman_filter_step(
      state.z, hit_z, hit_y, Velo::Tracking::param_w, state.y, state.ty, state.c11, state.c31, state.c33);

    // update z (note done in the filter, since needed only once)
    state.z = hit_z;
  }

  // add the noise at the last hit
  state.c22 += noise2PerLayer;
  state.c33 += noise2PerLayer;

  // finally, store the state
  return state;
}

__global__ void velo_kalman_fit(
  int* dev_atomics_velo,
  uint* dev_velo_track_hit_number,
  char* dev_velo_track_hits,
  char* dev_velo_states,
  char* dev_velo_kalman_beamline_states);

ALGORITHM(
  velo_kalman_fit,
  velo_kalman_fit_t,
  ARGUMENTS(
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_velo_states,
    dev_velo_kalman_beamline_states))
