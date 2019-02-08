#pragma once

#include "ConsolidatedTypes.cuh"
#include "SciFiEventModel.cuh"
#include <stdint.h>

namespace SciFi {
  namespace Consolidated {

    // Consolidated hits SoA.
    struct Hits : BaseHits {
      __device__ __host__ Hits(
        char* base_pointer,
        const uint track_offset,
        const uint total_number_of_hits,
        const SciFiGeometry* param_geom,
        const float* param_dev_inv_clus_res)
      {
        x0 = reinterpret_cast<float*>(base_pointer);
        z0 = reinterpret_cast<float*>(base_pointer + sizeof(float) * total_number_of_hits);
        m_endPointY = reinterpret_cast<float*>(base_pointer + sizeof(float) * 2 * total_number_of_hits);
        channel = reinterpret_cast<uint32_t*>(base_pointer + sizeof(float) * 3 * total_number_of_hits);
        assembled_datatype = reinterpret_cast<uint32_t*>(base_pointer + sizeof(float) * 4 * total_number_of_hits);

        x0 += track_offset;
        z0 += track_offset;
        m_endPointY += track_offset;
        channel += track_offset;
        assembled_datatype += track_offset;

        geom = param_geom;
        dev_inv_clus_res = param_dev_inv_clus_res;
      }

      __device__ __host__ SciFi::Hit get(const uint hit_number) const
      {
        return SciFi::Hit {x0[hit_number], z0[hit_number], m_endPointY[hit_number], channel[hit_number]};
      }

      __device__ __host__ uint32_t LHCbID(uint32_t index) const { return (10u << 28) + channel[index]; };
    };

    //----------------------------------------------------------------------
    // Struct for holding consolidated SciFi track information.
    struct Tracks : public ::Consolidated::Tracks {
      // Indices of associated UT tracks.
      uint* ut_track;

      float* qop;
      MiniState* states;

      __device__ __host__ Tracks(
        uint* atomics_base_pointer,
        uint* track_hit_number_base_pointer,
        float* qop_base_pointer,
        MiniState* states_base_pointer,
        uint* ut_track_base_pointer,
        const uint current_event_number,
        const uint number_of_events) :
        ::Consolidated::Tracks(
          atomics_base_pointer,
          track_hit_number_base_pointer,
          current_event_number,
          number_of_events)
      {
        ut_track = ut_track_base_pointer + tracks_offset(current_event_number);
        qop = qop_base_pointer + tracks_offset(current_event_number);
        states = states_base_pointer + tracks_offset(current_event_number);
      }

      __device__ __host__ Hits get_hits(
        char* hits_base_pointer,
        const uint track_number,
        const SciFiGeometry* scifi_geometry,
        const float* dev_inv_clus_res) const
      {
        return Hits {
          hits_base_pointer, track_offset(track_number), total_number_of_hits, scifi_geometry, dev_inv_clus_res};
      }
    }; // namespace Consolidated

  } // namespace Consolidated
} // end namespace SciFi
