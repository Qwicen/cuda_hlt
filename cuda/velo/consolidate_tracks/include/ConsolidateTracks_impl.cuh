template<bool mc_check_enabled>
__device__ Track<mc_check_enabled> createTrack(
  const TrackHits &track,
  const float* hit_Xs,
  const float* hit_Ys,
  const float* hit_Zs,
  const uint32_t* hit_IDs
) {
  Track<mc_check_enabled> t;
  for (int i = 0; i < track.hitsNum; ++i) {
    const auto hit_index = track.hits[i];
    Hit<mc_check_enabled> hit;
#ifdef MC_CHECK
    hit = {hit_Xs[hit_index],
      hit_Ys[hit_index],
      hit_Zs[hit_index],
      hit_IDs[hit_index]
    };
#else
    hit = {hit_Xs[hit_index],
      hit_Ys[hit_index],
      hit_Zs[hit_index]
    };
#endif
    t.addHit(hit);
  }
  return t;
}

template<bool mc_check_enabled>
__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const TrackHits* dev_tracks,
  Track <mc_check_enabled> * dev_output_tracks,
  uint32_t* dev_velo_cluster_container,
  uint* dev_module_cluster_start,
  uint* dev_module_cluster_num
) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const TrackHits* event_tracks = dev_tracks + event_number * VeloTracking::max_tracks;
  int* tracks_insert_pointer = dev_atomics_storage + event_number;
  const int number_of_tracks = *tracks_insert_pointer;
  
  // Due to dev_atomics_storage
  __syncthreads();

  if (threadIdx.x==0) {
    *tracks_insert_pointer = 0;
  }

  // Store accumulated tracks after the number of tracks
  // DvB: reusing the previous space for the weak tracks counter,
  // but storing in SoA rather than AoS now
  int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  auto accumulated_tracks = accumulated_tracks_base_pointer[event_number];
  // accumulated_tracks_base_pointer[event_number] = accumulated_tracks;
    
  // Pointers to data within event
  const uint number_of_hits = dev_module_cluster_start[VeloTracking::n_modules * number_of_events];
  const uint* module_hitStarts = dev_module_cluster_start + event_number * VeloTracking::n_modules;
  const uint hit_offset = module_hitStarts[0];
  
  // Order has changed since SortByPhi
  const float* hit_Ys   = (float*) (dev_velo_cluster_container + hit_offset);
  const float* hit_Zs   = (float*) (dev_velo_cluster_container + number_of_hits + hit_offset);
  const float* hit_Xs   = (float*) (dev_velo_cluster_container + 5 * number_of_hits + hit_offset);
  const uint32_t* hit_IDs  = (uint32_t*) (dev_velo_cluster_container + 2 * number_of_hits + hit_offset);
  
  // Due to tracks_insert_pointer
  __syncthreads();

  // Consolidate tracks in dev_output_tracks
  Track<mc_check_enabled> * destination_tracks = dev_output_tracks + accumulated_tracks;
  for (uint j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
    const uint element = j * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      const TrackHits track = event_tracks[element];

      VeloState state;
      TrackFitParameters parameters;
      b_means_square_fit(
        hit_Xs,
        hit_Ys,
        hit_Zs,
        track,
        parameters,
        state
      );

      // Require chi2 under threshold
      if ((track.hitsNum == 3 && state.chi2 < VeloTracking::max_chi2) || 
        (track.hitsNum > 3)) {
        const auto trackP = atomicAdd(tracks_insert_pointer, 1);

        Track<mc_check_enabled> t = createTrack<mc_check_enabled>(track, hit_Xs, hit_Ys, hit_Zs, hit_IDs);
        destination_tracks[trackP] = t;
      }
    }
  }
}
