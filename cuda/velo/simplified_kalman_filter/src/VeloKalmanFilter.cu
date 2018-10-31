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
  const Velo::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  uint* dev_velo_track_hits,
  uint* dev_velo_states,
  const Velo::State& stateAtBeamLine
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
    Velo::State beam_state = simplified_fit<true>(
      consolidated_hits,
      hit_Xs,
      hit_Ys,
      hit_Zs,
      hit_IDs,
      track,
      track.hitsNum,
      stateAtBeamLine
    );

    velo_states.set(event_tracks_offset + i, beam_state);
  }
}
