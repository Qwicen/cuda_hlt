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
  uint* dev_velo_track_hit_number,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states,
  VeloState* dev_kal_velo_states,
  const VeloTracking::TrackHits* dev_tracks
) {

  // one event per block -> block we look at one event -> threads look at different tracks inside event
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  
  const VeloTracking::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  

  //get number of accumulated tracks for one event
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const auto accumulated_tracks = accumulated_tracks_base_pointer[event_number];


  //get pointer to trackhits of current event
  int* tracks_insert_pointer = dev_atomics_storage + event_number;

  //get total number of tracks
  const int number_of_tracks = *tracks_insert_pointer;

  //each thread looks at multiple tracks -> for loop distributes tracks on threads
  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {

    // element is offset so that thread looks at right track
    const uint element = i * blockDim.x + threadIdx.x;

    //check that we still have tracks
    if (element < number_of_tracks) {
      const VeloTracking::TrackHits track = event_tracks[element];
      



      const VeloTracking::Hit<mc_check_enabled>* velo_track_hits = dev_velo_track_hits +
      dev_velo_track_hit_number[accumulated_tracks + element];


      //acumulated tracks gives the number of tracks in all previous events-> element gives position in current event
      VeloState * state_pointer = dev_velo_states + 2*accumulated_tracks + 2*element +1 ;


//velo_states = dev_velo_states + accumulated_tracks;
      const VeloState first = (dev_velo_states + accumulated_tracks)[element];

      
      simplified_fit<true>(        velo_track_hits,        first,        state_pointer,        track.hitsNum    );
      simplified_fit<true>(        velo_track_hits,        first,        dev_kal_velo_states + accumulated_tracks + element ,        track.hitsNum    );
      

    }
  }



}
