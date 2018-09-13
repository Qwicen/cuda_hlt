#pragma once

#include <cassert>
#include "cuda_runtime.h"
#include "VeloEventModel.cuh"

namespace Velo {
namespace Consolidated {

struct Hits {
  // SOA of all hits
  float* x;
  float* y;
  float* z;
  uint* LHCbID;

  __device__ __host__ Hits(const Hits& hits) : x(hits.x), y(hits.y),
    z(hits.z), LHCbID(hits.LHCbID) {}
  
  __device__ __host__ Hits(
    uint* base_pointer,
    const uint track_offset,
    const uint total_number_of_hits
  ) {
    x = reinterpret_cast<float*>(base_pointer + track_offset);
    y = reinterpret_cast<float*>(base_pointer + total_number_of_hits + track_offset);
    z = reinterpret_cast<float*>(base_pointer + 2*total_number_of_hits + track_offset);
    LHCbID = reinterpret_cast<uint*>(base_pointer + 3*total_number_of_hits + track_offset);
  }

  __device__ __host__ void set(
    const uint hit_number,
    const Velo::Hit& hit
  ) {
    x[hit_number] = hit.x;
    y[hit_number] = hit.y;
    z[hit_number] = hit.z;
    LHCbID[hit_number] = hit.LHCbID;
  }

  __device__ __host__ Velo::Hit get(
    const uint hit_number
  ) const {
    return Velo::Hit {
      x[hit_number],
      y[hit_number],
      z[hit_number],
      LHCbID[hit_number]
    };
  }
};

struct TracksDescription {
  // Prefix sum of all Velo track sizes
  uint* event_number_of_tracks;
  uint* event_tracks_offsets;
  uint total_number_of_tracks;
  uint number_of_events;

  __device__ __host__ TracksDescription(
    uint* base_pointer,
    const uint param_number_of_events
  ) : event_number_of_tracks(base_pointer),
    event_tracks_offsets(base_pointer + param_number_of_events),
    number_of_events(param_number_of_events) {
    total_number_of_tracks = event_tracks_offsets[number_of_events];
  }

  __device__ __host__ uint number_of_tracks(const uint event_number) const {
    assert(event_number < number_of_events);
    return event_number_of_tracks[event_number];
  }

  __device__ __host__ uint tracks_offset(const uint event_number) const {
    assert(event_number <= number_of_events);
    return event_tracks_offsets[event_number];
  }
};

struct Tracks : public TracksDescription {
  uint* track_number_of_hits;
  uint total_number_of_hits;
  
  __device__ __host__ Tracks(
    uint* atomics_base_pointer,
    uint* track_hit_number_base_pointer,
    const uint current_event_number,
    const uint number_of_events
  ) : TracksDescription(atomics_base_pointer, number_of_events) {
    track_number_of_hits = track_hit_number_base_pointer + tracks_offset(current_event_number);
    total_number_of_hits = *(track_hit_number_base_pointer + tracks_offset(number_of_events));
  }

  __device__ __host__ uint track_offset(const uint track_number) const {
    assert(track_number <= total_number_of_tracks);
    return track_number_of_hits[track_number];
  }

  __device__ __host__ uint number_of_hits(const uint track_number) const {
    assert(track_number < total_number_of_tracks);
    return track_number_of_hits[track_number+1] - track_number_of_hits[track_number];
  }

  __device__ __host__ Hits get_hits(uint* hits_base_pointer, const uint track_number) const {
    return Hits {hits_base_pointer, track_offset(track_number), total_number_of_hits};
  }
};

struct States {
  // SOA of Velo states
  float* x;
  float* y;
  float* tx;
  float* ty;
  
  float* c00;
  float* c20;
  float* c22;
  float* c11;
  float* c31;
  float* c33;

  float* chi2;
  float* z;
  bool* backward;

  __device__ __host__ States(
    uint* base_pointer,
    const uint total_number_of_tracks
  ) {
    x = reinterpret_cast<float*>(base_pointer);
    y = reinterpret_cast<float*>(base_pointer + total_number_of_tracks);
    tx = reinterpret_cast<float*>(base_pointer + 2*total_number_of_tracks);
    ty = reinterpret_cast<float*>(base_pointer + 3*total_number_of_tracks);
    c00 = reinterpret_cast<float*>(base_pointer + 4*total_number_of_tracks);
    c20 = reinterpret_cast<float*>(base_pointer + 5*total_number_of_tracks);
    c22 = reinterpret_cast<float*>(base_pointer + 6*total_number_of_tracks);
    c11 = reinterpret_cast<float*>(base_pointer + 7*total_number_of_tracks);
    c31 = reinterpret_cast<float*>(base_pointer + 8*total_number_of_tracks);
    c33 = reinterpret_cast<float*>(base_pointer + 9*total_number_of_tracks);
    chi2 = reinterpret_cast<float*>(base_pointer + 10*total_number_of_tracks);
    z = reinterpret_cast<float*>(base_pointer + 11*total_number_of_tracks);
    backward = reinterpret_cast<bool*>(base_pointer + 12*total_number_of_tracks);
  }

  __device__ __host__ void set(
    const uint track_number,
    const Velo::State& state
  ) {
    x[track_number] = state.x;
    y[track_number] = state.y;
    tx[track_number] = state.tx;
    ty[track_number] = state.ty;

    c00[track_number] = state.c00;
    c20[track_number] = state.c20;
    c22[track_number] = state.c22;
    c11[track_number] = state.c11;
    c31[track_number] = state.c31;
    c33[track_number] = state.c33;

    chi2[track_number] = state.chi2;
    z[track_number] = state.z;
    backward[track_number] = state.backward;
  }

  __device__ __host__ Velo::State get(
    const uint track_number
  ) const {
    Velo::State state;
    
    state.x = x[track_number];
    state.y = y[track_number];
    state.tx = tx[track_number];
    state.ty = ty[track_number];

    state.c00 = c00[track_number];
    state.c20 = c20[track_number];
    state.c22 = c22[track_number];
    state.c11 = c11[track_number];
    state.c31 = c31[track_number];
    state.c33 = c33[track_number];

    state.chi2 = chi2[track_number];
    state.z = z[track_number];
    state.backward = backward[track_number];

    return state;
  }
};

}
}
