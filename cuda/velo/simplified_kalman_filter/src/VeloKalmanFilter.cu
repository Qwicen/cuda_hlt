#include "../include/VeloKalmanFilter.cuh"

/**
 * @brief Helper function to filter one hit
 */
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
) {
  // compute the prediction
  const float dz = zhit - z;
  const float predx = x + dz * tx;

  const float dz_t_covTxTx = dz * covTxTx;
  const float predcovXTx = covXTx + dz_t_covTxTx;
  const float dx_t_covXTx = dz * covXTx;

  const float predcovXX = covXX + 2 * dx_t_covXTx + dz * dz_t_covTxTx;
  const float predcovTxTx = covTxTx;
  // compute the gain matrix
  const float R = 1.0f / ((1.0f / whit) + predcovXX);
  const float Kx = predcovXX * R;
  const float KTx = predcovXTx * R;
  // update the state vector
  const float r = xhit - predx;
  x = predx + Kx * r;
  tx = tx + KTx * r;
  // update the covariance matrix. we can write it in many ways ...
  covXX /*= predcovXX  - Kx * predcovXX */ = (1 - Kx) * predcovXX;
  covXTx /*= predcovXTx - predcovXX * predcovXTx / R */ = (1 - Kx) * predcovXTx;
  covTxTx = predcovTxTx - KTx * predcovXTx;
  // return the chi2
  return r * r * R;
}

__global__ void velo_fit(
  int* dev_atomics_storage,
  const VeloTracking::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states
) {

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const auto accumulated_tracks = accumulated_tracks_base_pointer[event_number];

  const VeloTracking::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  int* tracks_insert_pointer = dev_atomics_storage + event_number;
  const int number_of_tracks = *tracks_insert_pointer;

  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {
    const uint element = i * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      
      const VeloTracking::TrackHits track = event_tracks[element];

      const VeloTracking::Hit<mc_check_enabled>* velo_track_hits = dev_velo_track_hits +
      dev_velo_track_hit_number[accumulated_tracks + element];

      simplified_fit<false>(        velo_track_hits,        dev_velo_states[0],        dev_velo_states,        track      );
    }
  }



}
