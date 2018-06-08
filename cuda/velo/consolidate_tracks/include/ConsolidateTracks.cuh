#pragma once

#include "../../common/include/VeloDefinitions.cuh"
#include <stdint.h>

template<bool do_mc_check>
__device__ VeloTracking::Track <do_mc_check> createTrack(
  const VeloTracking::TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
) {

  VeloTracking::Track <do_mc_check> t;
  for ( int i = 0; i < track.hitsNum; ++i ) {
    const auto hit_index = track.hits[i];
    VeloTracking::Hit <do_mc_check> hit;
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
  }
  t.backward = t.hits[ t.hitsNum-1 ].z > t.hits[ 0 ].z;
  
  return t;
}

template <bool do_mc_check>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const VeloTracking::TrackHits* dev_tracks,
  VeloTracking::Track <do_mc_check> * dev_output_tracks,
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
  VeloTracking::Track <do_mc_check> * destination_tracks = dev_output_tracks + accumulated_tracks;
  // only write out one state: at the last measurement
  VeloState* destination_velo_states = dev_velo_states_out + accumulated_tracks;
  /* don't do consolidation now -> easier to check tracks offline */
  //Track <do_mc_check> * destination_tracks = dev_output_tracks + event_number * VeloTracking::max_tracks;
  for (unsigned int j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
    const unsigned int element = j * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      const VeloTracking::TrackHits track = event_tracks[element];
      VeloTracking::Track <do_mc_check> t = createTrack <do_mc_check> ( track, hit_Xs, hit_Ys, hit_Zs, hit_IDs );
      destination_tracks[element] = t;
      // state at last measurement is stored in second element of velo_states
      int state_index = VeloTracking::states_per_track * element + 1; 
      
      destination_velo_states[element] = velo_states[state_index];
      // if ( (threadIdx.x == 0 || threadIdx.x == 2 || threadIdx.x == 3 || threadIdx.x == 4) && blockIdx.x == 0 ) {
      // 	printf("state x = %f, z = %f \n", destination_velo_states[element].x, destination_velo_states[element].z);
      // 	printf("track numhits = %u \n", track.hitsNum);
      // }
    }
  }
}
