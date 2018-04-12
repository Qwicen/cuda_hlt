#include "../include/ConsolidateTracks.cuh"

__global__ void consolidate_tracks(
  int* dev_atomics_storage,
  const Track* dev_tracks,
  Track* dev_output_tracks,
  unsigned int* dev_hit_offsets,
  unsigned short* dev_hit_permutation
) {
  const unsigned int number_of_events = gridDim.x;
  const unsigned int event_number = blockIdx.x;

  unsigned int accumulated_tracks = 0;
  const Track* event_tracks = dev_tracks + event_number * MAX_TRACKS;

  const unsigned int hit_offset = dev_hit_offsets[event_number];
  const unsigned short* hit_permutation = dev_hit_permutation + hit_offset;

  // Obtain accumulated tracks
  for (unsigned int i=0; i<event_number; ++i) {
    const unsigned int number_of_tracks = dev_atomics_storage[i];
    accumulated_tracks += number_of_tracks;
  }

  // Store accumulated tracks after the number of tracks
  int* accumulated_tracks_base_pointer = dev_atomics_storage + number_of_events;
  accumulated_tracks_base_pointer[event_number] = accumulated_tracks;

  // Consolidate tracks in dev_output_tracks
  const unsigned int number_of_tracks = dev_atomics_storage[event_number];
  Track* destination_tracks = dev_output_tracks + accumulated_tracks;
  for (unsigned int j=0; j<(number_of_tracks + blockDim.x - 1) / blockDim.x; ++j) {
    const unsigned int element = j * blockDim.x + threadIdx.x;
    if (element < number_of_tracks) {
      Track t = event_tracks[element];

      for (int hit_no=0; hit_no<t.hitsNum; ++hit_no) {
        t.hits[hit_no] = hit_permutation[t.hits[hit_no]];
      }

      destination_tracks[element] = t;
    }
  }
}
