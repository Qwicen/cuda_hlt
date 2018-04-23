#include "VeloKalmanFilter.cuh"

/**
 * @brief Calculates the parameters according to a root means square fit
 */
__device__ void means_square_fit(
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const Track& track,
  TrackFitParameters& parameters,
  VeloState* velo_state
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
    
    const auto wx = PARAM_W;
    const auto wx_t_x = wx * x;
    const auto wx_t_z = wx * z;
    s0 += wx;
    sx += wx_t_x;
    sz += wx_t_z;
    sxz += wx_t_x * z;
    sz2 += wx_t_z * z;

    const auto wy = PARAM_W;
    const auto wy_t_y = wy * y;
    const auto wy_t_z = wy * z;
    u0 += wy;
    uy += wy_t_y;
    uz += wy_t_z;
    uyz += wy_t_y * z;
    uz2 += wy_t_z * z;
  }

  VeloState state;
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
      
      ch += dx * dx * PARAM_W + dy * dy * PARAM_W;

      // Nice :)
      // TODO: We can get rid of the X and Y read here
      // float sum_w_xzi_2 = CL_PARAM_W * x; // for each hit
      // float sum_w_xi_2 = CL_PARAM_W * hit_Xs[hitno]; // for each hit
      // ch = (sum_w_xzi_2 - sum_w_xi_2) + (sum_w_yzi_2 - sum_w_yi_2);

      nDoF += 2;
    }
    state.chi2 = ch / nDoF; 
  }

  state.x = state.x + state.tx * state.z;
  state.y = state.y + state.ty * state.z;

  // Store state at beam line
  *velo_state = state;

  // Keep some parameters around for the upcoming Kalman fit
  parameters.tx = state.tx;
  parameters.ty = state.ty;
}

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
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const Track& track,
  const TrackFitParameters& parameters,
  VeloState* velo_state
) {
  const int direction = (parameters.backward ? 1 : -1) * (upstream ? 1 : -1);
  const float noise2PerLayer = 1e-8 + 7e-6 * (parameters.tx * parameters.tx + parameters.ty * parameters.ty);

  // assume the hits are sorted,
  // but don't assume anything on the direction of sorting
  int firsthit = 0;
  int lasthit = track.hitsNum - 1;
  int dhit = 1;
  if ((hit_Zs[track.hits[lasthit]] - hit_Zs[track.hits[firsthit]]) * direction < 0) {
    const int temp = firsthit;
    firsthit = lasthit;
    lasthit = temp;
    dhit = -1;
  }

  // We filter x and y simultaneously but take them uncorrelated.
  // filter first the first hit.
  const auto hitno = track.hits[firsthit];
  VeloState state;
  state.x = hit_Xs[hitno];
  state.y = hit_Ys[hitno];
  state.z = hit_Zs[hitno];
  state.tx = parameters.tx;
  state.ty = parameters.ty;

  // Initialize the covariance matrix
  state.c00 = PARAM_W_INVERTED;
  state.c11 = PARAM_W_INVERTED;
  state.c20 = 0.f;
  state.c31 = 0.f;
  state.c22 = 1.f;
  state.c33 = 1.f;

  // add remaining hits
  state.chi2 = 0.0f;
  for (uint i=firsthit + dhit; i!=lasthit + dhit; i+=dhit) {
    const uint hitno = track.hits[i];
    const auto hit_x = hit_Xs[hitno];
    const auto hit_y = hit_Ys[hitno];
    const auto hit_z = hit_Zs[hitno];
    
    // add the noise
    state.c22 += noise2PerLayer;
    state.c33 += noise2PerLayer;

    // filter X and filter Y
    state.chi2 += velo_kalman_filter_step(state.z, hit_z, hit_x, PARAM_W, state.x, state.tx, state.c00, state.c20, state.c22);
    state.chi2 += velo_kalman_filter_step(state.z, hit_z, hit_y, PARAM_W, state.y, state.ty, state.c11, state.c31, state.c33);
    
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
  const Track* dev_tracks,
  VeloState* dev_velo_states,
  const bool is_consolidated
) {
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;
  const uint tracks_offset = event_number * MAX_TRACKS;

  // Pointers to data within the event
  const uint number_of_hits = dev_module_cluster_start[52 * number_of_events];
  
  // Order has changed since SortByPhi
  const float* hit_Ys = (float*) (dev_velo_cluster_container);
  const float* hit_Zs = (float*) (dev_velo_cluster_container + number_of_hits);
  const float* hit_Xs = (float*) (dev_velo_cluster_container + 5 * number_of_hits);

  // Reconstructed tracks
  const Track* tracks = dev_tracks + tracks_offset;
  const uint number_of_tracks = dev_atomics_storage[event_number];
  VeloState* velo_states = dev_velo_states;

  // The location of the track depends on whether the consolidation took place
  if (is_consolidated) {
    const uint track_start = dev_atomics_storage[number_of_events + event_number];
    velo_states += track_start * STATES_PER_TRACK;
  } else {
    velo_states += event_number * MAX_TRACKS * STATES_PER_TRACK;
  }

  // Iterate over the tracks and calculate fits
  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {
    const auto element = i * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      // Base pointer to velo_states for this element
      VeloState* velo_state_base = velo_states + element * STATES_PER_TRACK;

      // Means square fit
      const auto track = tracks[element];
      TrackFitParameters parameters;

      // State at beam line
      means_square_fit(
        hit_Xs,
        hit_Ys,
        hit_Zs,
        track,
        parameters,
        velo_state_base
      );

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
        hit_Xs,
        hit_Ys,
        hit_Zs,
        track,
        parameters,
        velo_state_base + 1
      );

      // Upstream fit
      simplified_fit<true>(
        hit_Xs,
        hit_Ys,
        hit_Zs,
        track,
        parameters,
        velo_state_base + 2
      );
    }
  }
}
