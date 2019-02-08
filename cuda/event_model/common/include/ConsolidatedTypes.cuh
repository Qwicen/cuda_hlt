#pragma once

#include <cassert>
#include "cuda_runtime.h"
#include "Common.h"

namespace Consolidated {

  // base_pointer contains first: an array with the number of tracks in every event
  // second: an array with offsets to the tracks for every event
  struct TracksDescription {
    // Prefix sum of all Velo track sizes
    uint* event_number_of_tracks;
    uint* event_tracks_offsets;
    uint total_number_of_tracks;
    uint number_of_events;

    __device__ __host__ TracksDescription(uint* base_pointer, const uint param_number_of_events) :
      event_number_of_tracks(base_pointer), event_tracks_offsets(base_pointer + param_number_of_events),
      number_of_events(param_number_of_events)
    {
      total_number_of_tracks = event_tracks_offsets[number_of_events];
    }

    __device__ __host__ uint number_of_tracks(const uint event_number) const
    {
      assert(event_number < number_of_events);
      return event_number_of_tracks[event_number];
    }

    __device__ __host__ uint tracks_offset(const uint event_number) const
    {
      assert(event_number <= number_of_events);
      return event_tracks_offsets[event_number];
    }
  };

  // atomics_base_pointer size needed: 2 * number_of_events
  struct Tracks : public TracksDescription {
    uint* track_number_of_hits;
    uint total_number_of_hits;

    __device__ __host__ Tracks(
      uint* atomics_base_pointer,
      uint* track_hit_number_base_pointer,
      const uint current_event_number,
      const uint number_of_events) :
      TracksDescription(atomics_base_pointer, number_of_events)
    {
      track_number_of_hits = track_hit_number_base_pointer + tracks_offset(current_event_number);
      total_number_of_hits = *(track_hit_number_base_pointer + tracks_offset(number_of_events));
    }

    __device__ __host__ uint track_offset(const uint track_number) const
    {
      assert(track_number <= total_number_of_tracks);
      return track_number_of_hits[track_number];
    }

    __device__ __host__ uint number_of_hits(const uint track_number) const
    {
      assert(track_number < total_number_of_tracks);
      return track_number_of_hits[track_number + 1] - track_number_of_hits[track_number];
    }
  };

} // namespace Consolidated
