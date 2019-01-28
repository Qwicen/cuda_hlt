#pragma once

#include "ConsolidatedTypes.cuh"
#include "UTEventModel.cuh"
#include <stdint.h>

namespace UT {
namespace Consolidated {

// SoA of consolidated UT hits.
struct Hits {
  constexpr static uint number_of_arrays = 8;
  float* yBegin;
  float* yEnd;
  float* zAtYEq0;
  float* xAtYEq0;
  float* weight;
  uint32_t* LHCbID;
  uint8_t* plane_code;
  uint number_of_hits;

  __device__ __host__ Hits(const Hits& hits) :
    yBegin(hits.yBegin),
    yEnd(hits.yEnd),
    zAtYEq0(hits.zAtYEq0),
    xAtYEq0(hits.xAtYEq0),
    weight(hits.weight),
    LHCbID(hits.LHCbID),
    plane_code(hits.plane_code)
  {}

  __device__ __host__ Hits(char* base_pointer, const uint track_offset, const uint total_number_of_hits)
  {
    yBegin = reinterpret_cast<float*>(base_pointer);
    yEnd = reinterpret_cast<float*>(base_pointer + sizeof(uint32_t) * total_number_of_hits);
    zAtYEq0 = reinterpret_cast<float*>(base_pointer + sizeof(uint32_t) * 2 * total_number_of_hits);
    xAtYEq0 = reinterpret_cast<float*>(base_pointer + sizeof(uint32_t) * 3 * total_number_of_hits);
    weight = reinterpret_cast<float*>(base_pointer + sizeof(uint32_t) * 4 * total_number_of_hits);
    LHCbID = reinterpret_cast<uint32_t*>(base_pointer + sizeof(uint32_t) * 5 * total_number_of_hits);
    plane_code = reinterpret_cast<uint8_t*>(base_pointer + sizeof(uint32_t) * 6 * total_number_of_hits);
    
    yBegin += track_offset;
    yEnd += track_offset;
    zAtYEq0 += track_offset;
    xAtYEq0 += track_offset;
    weight += track_offset;
    LHCbID += track_offset;
    plane_code += track_offset;
    number_of_hits += track_offset;
  }

  __device__ __host__ void set(const uint hit_number, const UT::Hit& hit)
  {
    yBegin[hit_number] = hit.yBegin;
    yEnd[hit_number] = hit.yEnd;
    zAtYEq0[hit_number] = hit.zAtYEq0;
    xAtYEq0[hit_number] = hit.xAtYEq0;
    weight[hit_number] = hit.weight;
    LHCbID[hit_number] = hit.LHCbID;
    plane_code[hit_number] = hit.plane_code;
  }

  __device__ __host__ UT::Hit get(const uint hit_number) const
  {
    return UT::Hit {
      yBegin[hit_number],
      yEnd[hit_number],
      zAtYEq0[hit_number],
      xAtYEq0[hit_number],
      weight[hit_number],
      LHCbID[hit_number],
      plane_code[hit_number]
    };
  }
};

//----------------------------------------------------------------------
// Struct for holding VELO track information.
struct Tracks : public ::Consolidated::Tracks {

  // Indices of associated VELO tracks.
  uint* velo_track;

  // Array of q/p for each track.
  float* qop;

  __device__ __host__ Tracks(
    uint* atomics_base_pointer,
    uint* track_hit_number_base_pointer,
    float* qop_base_pointer,
    uint* velo_track_base_pointer,
    const uint current_event_number,
    const uint number_of_events) :
    ::Consolidated::Tracks(atomics_base_pointer, track_hit_number_base_pointer, current_event_number, number_of_events)
  {
    velo_track = velo_track_base_pointer + tracks_offset(current_event_number);
    qop = qop_base_pointer + tracks_offset(current_event_number);
  }

  __device__ __host__ Hits get_hits(char* hits_base_pointer, const uint track_number) const
  {
    return Hits {hits_base_pointer, track_offset(track_number), total_number_of_hits};
  }
};

} // end namespace Consolidated
} // end namespace UT
