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

/**
 * @brief Fit the track with a Kalman filter,
 *        allowing for some scattering at every hit
 */
template<bool upstream>
__device__ void simplified_fit(
  const VeloTracking::Track<mc_check_enabled>& track,
  const VeloState& stateAtBeamLine,
  VeloState* velo_state
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
  VeloState state;
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
      
      // Downstream fit
      simplified_fit<false>(
        track,
        stateAtBeamLine,
        velo_state_base + total_number_of_tracks
      );

      // Upstream fit
      simplified_fit<true>(
        track,
        stateAtBeamLine,
        velo_state_base + 2*total_number_of_tracks
      );
    }
  }
}
