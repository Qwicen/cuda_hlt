#include "../include/ConsolidateTracks.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  Track* dev_tracks,
  const unsigned int number_of_events
) {
  unsigned int accumulated_tracks = dev_atomics_storage;
  Track* destination_tracks = dev_tracks;

  for (unsigned int i=0; i<number_of_events; ++i) {
    const unsigned int number_of_tracks = dev_atomics_storage[i];
    const Track* event_tracks = dev_tracks + (i + 1) * MAX_TRACKS;
    for (unsigned int j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
      const unsigned int element = j * blockDim.x + threadIdx.x;
      if (element < number_of_tracks) {
        destination_tracks[element] = event_tracks[element];
      }
    }
    destination_tracks += number_of_tracks;
    __syncthreads();

    // Make dev_atomics_storage store accumulated tracks
    accumulated_tracks += number_of_tracks;
    dev_atomics_storage[i] = accumulated_tracks;
    
    __syncthreads();
  }
}
