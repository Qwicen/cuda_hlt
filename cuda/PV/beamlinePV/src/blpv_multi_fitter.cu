#include "blpv_multi_fitter.cuh"



__global__ void blpv_multi_fitter(
  int * dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices) {



  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const Velo::Consolidated::Tracks velo_tracks {(uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};
  
  const uint number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  //to get started one thread per peak


/*
    for(int i = 0; i < number_of_tracks_event/blockDim.x + 1; i++) {
    int index = blockDim.x * i + threadIdx.x;
    if(index < number_of_tracks_event) { }
  }
*/
}