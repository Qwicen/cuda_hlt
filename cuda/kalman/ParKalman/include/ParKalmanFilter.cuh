#pragma once

#include "KalmanParametrizations.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "ParKalmanMethods.cuh"

#include "SciFiConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"

#include "MiniState.cuh"
#include "SciFiDefinitions.cuh"

#include "Handler.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"
#include "ArgumentsKalmanFilter.cuh"

namespace ParKalmanFilter {

  //----------------------------------------------------------------------
  // General method for updating states.
  __device__ void UpdateState(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_layers,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_layers,
    int forward,
    int i_hit,
    Vector5& x,
    SymMatrix5x5& C,
    KalmanFloat& lastz,
    trackInfo& tI);

  //----------------------------------------------------------------------
  // General method for predicting states.
  __device__ void PredictState(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_layers,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_layers,
    int forward,
    int i_hit,
    Vector5& x,
    SymMatrix5x5& C,
    KalmanFloat& lastz,
    trackInfo& tI);

  //----------------------------------------------------------------------
  // Forward fit iteration.
  __device__ void ForwardFit(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_layers,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_layers,
    Vector5& x,
    SymMatrix5x5& C,
    KalmanFloat& lastz,
    trackInfo& tI);

  //----------------------------------------------------------------------
  // Backward fit iteration.
  __device__ void BackwardFit(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_layers,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_layers,
    Vector5& x,
    SymMatrix5x5& C,
    KalmanFloat& lastz,
    trackInfo& tI);

  //----------------------------------------------------------------------
  // Create the output track.
  __device__ void MakeTrack(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_layers,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_layers,
    const Vector5& x,
    const SymMatrix5x5& C,
    const KalmanFloat& z,
    const trackInfo& tI,
    FittedTrack& track);

  //----------------------------------------------------------------------
  // Run the Kalman filter on a track.
  __device__ FittedTrack fit(
    const Velo::Consolidated::Hits& velo_hits,
    const uint n_velo_hits,
    const UT::Consolidated::Hits& ut_hits,
    const uint n_ut_hits,
    const SciFi::Consolidated::Hits& scifi_hits,
    const uint n_scifi_hits,
    const KalmanFloat init_qop,
    const KalmanParametrizations& kalman_params,
    FittedTrack& track);

} // namespace ParKalmanFilter

//----------------------------------------------------------------------
// Main execution of the parametrized Kalman Filter.
__global__ void KalmanFilter(
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  char* dev_velo_track_hits,
  int* dev_atomics_veloUT,
  uint* dev_ut_track_hit_number,
  char* dev_ut_consolidated_hits,
  float* dev_ut_qop,
  uint* dev_velo_indices,
  int* dev_n_scifi_tracks,
  uint* dev_scifi_track_hit_number,
  char* dev_scifi_consolidated_hits,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_ut_indices,
  ParKalmanFilter::FittedTrack* dev_kf_tracks,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params);

ALGORITHM(
  KalmanFilter,
  kalman_filter_t,
  ARGUMENTS(
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_velo_track_hits,
    dev_atomics_ut,
    dev_ut_track_hit_number,
    dev_ut_track_hits,
    dev_ut_qop,
    dev_ut_track_velo_indices,
    dev_atomics_scifi,
    dev_scifi_track_hit_number,
    dev_scifi_track_hits,
    dev_scifi_qop,
    dev_scifi_states,
    dev_scifi_track_ut_indices,
    dev_kf_tracks))
