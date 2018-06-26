#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include <stdint.h>

template<bool mc_check_enabled>
__device__ VeloTracking::Track <mc_check_enabled> createTrack(
  const VeloTracking::TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs,
  const VeloState& state_at_beam_line 
) {

  VeloTracking::Track <mc_check_enabled> t;
  for ( int i = 0; i < track.hitsNum; ++i ) {
    const auto hit_index = track.hits[i];
    VeloTracking::Hit <mc_check_enabled> hit;
#ifdef MC_CHECK
    hit = { hit_Xs[ hit_index ],
	    hit_Ys[ hit_index ],
	    hit_Zs[ hit_index ],
	    hit_IDs[ hit_index ]
    };
#else
    hit = { hit_Xs[ hit_index ],
	    hit_Ys[ hit_index ],
	    hit_Zs[ hit_index ]
    };
#endif
    t.addHit( hit );
    if ( i == 0 ) {
      t.backward = state_at_beam_line.z > hit.z;
      //t.first_z = hit.z;
    }
    // if ( i == track.hitsNum-1 )
    //   t.last_z = hit.z;
  }
  
  return t;
}

template <bool mc_check_enabled>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const VeloTracking::TrackHits* dev_tracks,
  VeloTracking::Track <mc_check_enabled> * dev_output_tracks,
  VeloState* dev_velo_states,
  VeloState* dev_velo_states_out,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num
) {
  const unsigned int number_of_events = gridDim.x;
  const unsigned int event_number = blockIdx.x;

  unsigned int accumulated_tracks = 0;
  const VeloTracking::TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  const VeloState* velo_states  = dev_velo_states + event_number * VeloTracking::max_tracks * VeloTracking::states_per_track;

  // Obtain accumulated tracks
  // DvB: this seems to be calculated on every thread of a block,
  // could probably be parallelized
  for (unsigned int i=0; i<event_number; ++i) {
    const unsigned int number_of_tracks = dev_atomics_storage[i];
    accumulated_tracks += number_of_tracks;
  }

  // Store accumulated tracks after the number of tracks
  // DvB: reusing the previous space for the weak tracks counter,
  // but storing in SoA rather than AoS now
  int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  accumulated_tracks_base_pointer[event_number] = accumulated_tracks;
    
  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs  = (uint32_t*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);

  
  // Consolidate tracks in dev_output_tracks
  const unsigned int number_of_tracks = dev_atomics_storage[event_number];
  VeloTracking::Track <mc_check_enabled> * destination_tracks = dev_output_tracks + accumulated_tracks;
  // only write out one state: at the last measurement
  VeloState* destination_velo_states = dev_velo_states_out + accumulated_tracks;
  /* don't do consolidation now -> easier to check tracks offline */
  //Track <mc_check_enabled> * destination_tracks = dev_output_tracks + event_number * VeloTracking::max_tracks;
  for (unsigned int j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
    const unsigned int element = j * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      // state at beam line is stored in first element,
      // state at end of Velo is stored in third element of velo_states
      int state_index = VeloTracking::states_per_track * element; 
      destination_velo_states[element] = velo_states[state_index];
      
      const VeloTracking::TrackHits track = event_tracks[element];
      VeloTracking::Track <mc_check_enabled> t = createTrack <mc_check_enabled> ( track, hit_Xs, hit_Ys, hit_Zs, hit_IDs, velo_states[ state_index ] );

      destination_tracks[element] = t;
      
    }
  }
}
