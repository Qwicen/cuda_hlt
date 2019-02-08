#pragma once

#include <stdint.h>
#include <cassert>
#include "States.cuh"
#include "VeloEventModel.cuh"
#include "ConsolidatedTypes.cuh"

namespace Velo {
  namespace Consolidated {

    struct Hits {
      // SOA of all hits
      float* x;
      float* y;
      float* z;
      uint* LHCbID;
      uint number_of_hits;

      __device__ __host__ Hits(const Hits& hits) : x(hits.x), y(hits.y), z(hits.z), LHCbID(hits.LHCbID) {}

      __device__ __host__ Hits(char* base_pointer, const uint track_offset, const uint total_number_of_hits)
      {
        x = reinterpret_cast<float*>(base_pointer);
        y = reinterpret_cast<float*>(base_pointer + sizeof(float) * total_number_of_hits);
        z = reinterpret_cast<float*>(base_pointer + sizeof(float) * 2 * total_number_of_hits);
        LHCbID = reinterpret_cast<uint*>(base_pointer + sizeof(float) * 3 * total_number_of_hits);

        x += track_offset;
        y += track_offset;
        z += track_offset;
        LHCbID += track_offset;
      }

      __device__ __host__ void set(const uint hit_number, const Velo::Hit& hit)
      {
        x[hit_number] = hit.x;
        y[hit_number] = hit.y;
        z[hit_number] = hit.z;
        LHCbID[hit_number] = hit.LHCbID;
      }

      __device__ __host__ Velo::Hit get(const uint hit_number) const
      {
        return Velo::Hit {x[hit_number], y[hit_number], z[hit_number], LHCbID[hit_number]};
      }
    };

    struct Tracks : public ::Consolidated::Tracks {
      __device__ __host__ Tracks(
        uint* atomics_base_pointer,
        uint* track_hit_number_base_pointer,
        const uint current_event_number,
        const uint number_of_events) :
        ::Consolidated::Tracks(
          atomics_base_pointer,
          track_hit_number_base_pointer,
          current_event_number,
          number_of_events)
      {}

      __device__ __host__ Hits get_hits(char* hits_base_pointer, const uint track_number) const
      {
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

      __device__ __host__ States(char* base_pointer, const uint total_number_of_tracks)
      {
        x = reinterpret_cast<float*>(base_pointer);
        y = reinterpret_cast<float*>(base_pointer + sizeof(float) * total_number_of_tracks);
        tx = reinterpret_cast<float*>(base_pointer + sizeof(float) * 2 * total_number_of_tracks);
        ty = reinterpret_cast<float*>(base_pointer + sizeof(float) * 3 * total_number_of_tracks);
        c00 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 4 * total_number_of_tracks);
        c20 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 5 * total_number_of_tracks);
        c22 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 6 * total_number_of_tracks);
        c11 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 7 * total_number_of_tracks);
        c31 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 8 * total_number_of_tracks);
        c33 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 9 * total_number_of_tracks);
        chi2 = reinterpret_cast<float*>(base_pointer + sizeof(float) * 10 * total_number_of_tracks);
        z = reinterpret_cast<float*>(base_pointer + sizeof(float) * 11 * total_number_of_tracks);
        backward = reinterpret_cast<bool*>(base_pointer + sizeof(float) * 12 * total_number_of_tracks);
      }

      __device__ __host__ void set(const uint track_number, const VeloState& state)
      {
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

      __device__ __host__ VeloState get(const uint track_number) const
      {
        VeloState state;

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

  } // namespace Consolidated
} // namespace Velo
