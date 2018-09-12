#pragma once

#include "cuda_runtime.h"
#include "VeloEventModel.cuh"

namespace Consolidated {
namespace Velo {

struct Tracks {
  // Prefix sum of all Velo track sizes
  uint* track_hit_number;

  __device__ __host__ Tracks() {}
  __device__ __host__ Tracks(uint* track_hit_number)
    : track_hit_number(track_hit_number) {}
};

struct Hits {
  // SOA of all hits
  float* x;
  float* y;
  float* z;
  float* ID;

  
  __device__ __host__ Hits(uint* base_pointer,
    const uint track_hit_number,
    const uint total_number_of_hits
  ) {
    x = reinterpret_cast<float*>(base_pointer + track_hit_number);
    y = reinterpret_cast<float*>(base_pointer + total_number_of_hits + track_hit_number);
    z = reinterpret_cast<float*>(base_pointer + 2*total_number_of_hits + track_hit_number);
    ID = reinterpret_cast<float*>(base_pointer + 3*total_number_of_hits + track_hit_number);
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

  __device__ __host__ States() {}
  __device__ __host__ States(
    uint* base_pointer,
    const uint track_number,
    const uint total_number_of_tracks
  ) {
    x = reinterpret_cast<float*>(base_pointer + track_number);
    y = reinterpret_cast<float*>(base_pointer + total_number_of_tracks + track_number);
    tx = reinterpret_cast<float*>(base_pointer + 2*total_number_of_tracks + track_number);
    ty = reinterpret_cast<float*>(base_pointer + 3*total_number_of_tracks + track_number);
    c00 = reinterpret_cast<float*>(base_pointer + 4*total_number_of_tracks + track_number);
    c20 = reinterpret_cast<float*>(base_pointer + 5*total_number_of_tracks + track_number);
    c22 = reinterpret_cast<float*>(base_pointer + 6*total_number_of_tracks + track_number);
    c11 = reinterpret_cast<float*>(base_pointer + 7*total_number_of_tracks + track_number);
    c31 = reinterpret_cast<float*>(base_pointer + 8*total_number_of_tracks + track_number);
    c33 = reinterpret_cast<float*>(base_pointer + 9*total_number_of_tracks + track_number);
    chi2 = reinterpret_cast<float*>(base_pointer + 10*total_number_of_tracks + track_number);
    z = reinterpret_cast<float*>(base_pointer + 11*total_number_of_tracks + track_number);
    bool* char_base_pointer = reinterpret_cast<bool*>(base_pointer + 12*total_number_of_tracks);
    // Note: The operator+ below behaves differently from the ones above, as intended.
    backward = char_base_pointer + track_number;
  }

  __device__ __host__ void set(const ::Velo::State& state) {
    *x = state.x;
    *y = state.y;
    *tx = state.tx;
    *ty = state.ty;

    *c00 = state.c00;
    *c20 = state.c20;
    *c22 = state.c22;
    *c11 = state.c11;
    *c31 = state.c31;
    *c33 = state.c33;

    *chi2 = state.chi2;
    *z = state.z;
    *backward = state.backward;
  }
};

}
}
