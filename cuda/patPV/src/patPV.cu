#include "patPV.cuh"

__global__ void patPV(
  int* dev_atomics_storage,
  const VeloTracking::TrackHits* dev_tracks,
  uint* dev_velo_track_hit_number,
  uint* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num,
  VeloTracking::Hit<mc_check_enabled>* dev_velo_track_hits,
  VeloState* dev_velo_states
) {


  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const VeloTracking::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  int* tracks_insert_pointer = dev_atomics_storage + event_number;
  const int number_of_tracks = *tracks_insert_pointer;

  // Store accumulated tracks after the number of tracks
  // Reusing the previous space for the weak tracks counter,
  // but storing in SoA rather than AoS now
  const int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  const auto accumulated_tracks = accumulated_tracks_base_pointer[event_number];
    
  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs = (uint32_t*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);

  // Consolidate tracks in dev_output_tracks
  VeloState* velo_states = dev_velo_states + accumulated_tracks;

  for (uint i=0; i<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++i) {
    const uint element = i * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      const VeloTracking::TrackHits track = event_tracks[element];

      VeloState* velo_state_base = velo_states + element;
      const auto stateAtBeamLine = *velo_state_base;
      // Pointer to velo_track_hit container
      VeloTracking::Hit<mc_check_enabled>* velo_track_hits = dev_velo_track_hits +
        dev_velo_track_hit_number[accumulated_tracks + element];

      // Store X, Y, Z and ID in consolidated container
      for (uint j=0; j<track.hitsNum; ++j) {
        const auto hit_index = track.hits[j];
        velo_track_hits[j] = VeloTracking::Hit<mc_check_enabled>{
          hit_Xs[hit_index],
          hit_Ys[hit_index],
          hit_Zs[hit_index]
#ifdef MC_CHECK
          ,hit_IDs[hit_index]
#endif
        };
      }

      // Calculate and store fit in consolidated container
      velo_states[element] = simplified_fit2<false>(        velo_track_hits,        track , stateAtBeamLine   );

    }
  }




};
