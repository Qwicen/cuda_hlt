#pragma once

// A table of PV index inside an event for a track at global index and
// the values used when calculating the association.

#include <stdint.h>
#include <cassert>
#include <States.cuh>
#include <VeloEventModel.cuh>
#include <ConsolidatedTypes.cuh>

namespace Associate {
namespace Consolidated {

struct EventTable {
  // SOA of associated indices and values
  uint* pv = nullptr;
  float* value = nullptr;
  uint const size = 0;

  __device__ __host__ EventTable(uint const s,
                                 uint* pv_pointer,
                                 float* value_pointer)
    : pv{pv_pointer}, value{value_pointer}, size{s} {}
};

struct Table {
  // SOA of associated indices and values
  uint* total_number = nullptr;
  float* cutoff_value = nullptr;
  uint* pv = nullptr;
  float* value = nullptr;

  __device__ __host__ Table(char* base_pointer,
                            uint const tn)
  {
    total_number = reinterpret_cast<uint*>(base_pointer);
    *total_number = tn;
    cutoff_value = reinterpret_cast<float*>(total_number + 1);
    pv = reinterpret_cast<uint*>(cutoff_value + 1);
    // pv is now a pointer to uint, not char
    value = reinterpret_cast<float*>(pv + tn);
  }

  __device__ __host__ EventTable event_table(::Consolidated::TracksDescription const& track_index,
                                             uint const event_number) const {
    uint const event_tracks_offset = track_index.tracks_offset(event_number);
    return {track_index.number_of_tracks(event_number),
            pv + event_tracks_offset,
            value + event_tracks_offset};
  }

  __device__ __host__ void set_cutoff(float const val) const {
    *cutoff_value = val;
  }

  __device__ __host__ float cutoff() const {
    return *cutoff_value;
  }

  __device__ __host__ static uint size(uint const tn) {
    return sizeof(uint) * (tn + 1) +  sizeof(float) * (tn + 1);
  }
};

}
}
