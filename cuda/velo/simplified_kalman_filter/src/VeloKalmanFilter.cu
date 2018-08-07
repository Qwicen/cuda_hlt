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
  const uint32_t* dev_velo_cluster_container,
  const uint* dev_module_cluster_start,
  const int* dev_atomics_storage,
  const VeloTracking::Track<mc_check_enabled>* dev_tracks,
  VeloState* dev_velo_states
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;

  // Reconstructed tracks
  const uint number_of_tracks = dev_atomics_storage[event_number];
  const uint track_start = dev_atomics_storage[number_of_events + event_number];
  const uint total_number_of_tracks = dev_atomics_storage[2*number_of_events];
  const VeloTracking::Track<mc_check_enabled>* tracks = dev_tracks + track_start;
  VeloState* velo_states = dev_velo_states + track_start;

  // Iterate over the tracks and calculate fits
  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {
    const auto element = i * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      // Pointer to current element in velo states
      VeloState* velo_state_base = velo_states + element;
      const auto track = tracks[element];

      // Fetch means square fit 
      const auto stateAtBeamLine = *velo_state_base;

      // Always calculate two simplified Kalman fits and store their results to VeloState:
      // Downstream and upstream
      //
      // Note:
      // Downstream is equivalent to
      // ((!backward && m_stateEndVeloKalmanFit) || m_addStateFirstLastMeasurementKalmanFit)
      // 
      // Upstream is equivalent to
      // (m_stateClosestToBeamKalmanFit || m_addStateFirstLastMeasurementKalmanFit)
      
      // Downstream fit (away from the interaction region: lowest z for backward tracks, highest z for forward tracks)
      simplified_fit<false>(
        track,
        stateAtBeamLine,
        velo_state_base + total_number_of_tracks
      );

      // Upstream fit (towards the interaction region: highest z for backward tracks, lowest z for forward tracks)
      simplified_fit<true>(
        track,
        stateAtBeamLine,
        velo_state_base + 2*total_number_of_tracks
      );
    }
  }
}
