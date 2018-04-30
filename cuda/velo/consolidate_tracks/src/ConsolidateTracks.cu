#include "../include/ConsolidateTracks.cuh"

template<bool mc_check>
__device__ Track <mc_check> createTrack(
  const TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
) {

  Track <mc_check> t;
  for ( int i = 0; i < track.hitsNum; ++i ) {
    const auto hit_index = track.hits[i];
    Hit <mc_check> hit;
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
  return t;
}

template <bool mc_check>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  Track <mc_check> * dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num
) {
  const unsigned int number_of_events = gridDim.x;
  const unsigned int event_number = blockIdx.x;

  unsigned int accumulated_tracks = 0;
  const TrackHits* event_tracks = dev_tracks + event_number * MAX_TRACKS;

  // Obtain accumulated tracks
  for (unsigned int i=0; i<event_number; ++i) {
    const unsigned int number_of_tracks = dev_atomics_storage[i];
    accumulated_tracks += number_of_tracks;
  }

  // Store accumulated tracks after the number of tracks
  int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  accumulated_tracks_base_pointer[event_number] = accumulated_tracks;

  
  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[N_MODULES * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * N_MODULES;
  const uint* module_hitNums = dev_module_cluster_num + event_number * N_MODULES;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs  = (uint32_t*) (dev_velo_cluster_container + 3 * number_of_hits + hit_offset);

  
  // Consolidate tracks in dev_output_tracks
  const unsigned int number_of_tracks = dev_atomics_storage[event_number];
  //Track* destination_tracks = dev_output_tracks + accumulated_tracks;
  /* don't do consolidation now -> easier to check tracks offline */
  Track <mc_check> * destination_tracks = dev_output_tracks + event_number * MAX_TRACKS;
  for (unsigned int j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
    const unsigned int element = j * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      const TrackHits track = event_tracks[element];
      Track <mc_check> t = createTrack <mc_check> ( track, hit_Xs, hit_Ys, hit_Zs, hit_IDs );
      destination_tracks[element] = t;
    }
  }
}
