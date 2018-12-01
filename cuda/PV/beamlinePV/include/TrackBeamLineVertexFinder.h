#pragma once

#include "Common.h"
#include "Logger.h"
#include "PV_Definitions.cuh"
#include "VeloConsolidated.cuh"
#include "float_operations.h"

#include <algorithm>
#include <math.h>

// structure with minimal track info needed for PV search
struct PVTrack {
  __host__ __device__ PVTrack() {}
  __host__ __device__ PVTrack(const VeloState& state, float dz, unsigned short _index) :
    z {float(state.z + dz)}, x {float(state.x + dz * state.tx), float(state.y + dz * state.ty)},
    tx {float(state.tx), float(state.ty)}, index {_index}, old_z {float(state.z)}
  {
    // perhaps we should invert it /before/ switching to single FPP?
    // it doesn't seem to make much difference.

    PV::myfloat state_tmp_c00 = state.c00;
    PV::myfloat state_tmp_c11 = state.c11;

    float dz2 = dz * dz;

    state_tmp_c00 += dz2 * state.c22 + 2.f * dz * state.c20;
    state_tmp_c11 += dz2 * state.c33 + 2.f * dz * state.c31;
    W_00 = 1.f / state_tmp_c00;
    W_11 = 1.f / state_tmp_c11;
  }
  float z {0};
  float old_z;
  float2 x;  /// position (x,y)
  float2 tx; /// direction (tx,ty)
  // to do: check whether this needs to be a double
  float W_00; /// weightmatrix
  float W_11;
  unsigned short index {0}; /// index in the list with tracks
};

template<typename FTYPE>
__host__ __device__ FTYPE sqr(FTYPE x)
{
  return x * x;
}

struct Extremum {
  __host__ __device__ Extremum(unsigned short _index, float _value, float _integral) :
    index {_index}, value {_value}, integral {_integral} {};
  __host__ __device__ Extremum() {};
  unsigned short index;
  float value;
  float integral;
};

struct Cluster {
  __host__ __device__ Cluster(unsigned short _izfirst, unsigned short _izlast, unsigned short _izmax) :
    izfirst {_izfirst}, izlast {_izlast}, izmax {_izmax}
  {}
  unsigned short izfirst;
  unsigned short izlast;
  unsigned short izmax;
  __host__ __device__ Cluster() {};
};

struct SeedZWithIteratorPair {
  using iterator = std::vector<PVTrack>::iterator;
  float z;
  iterator begin;
  iterator end;
  __host__ __device__ SeedZWithIteratorPair(float _z, iterator _begin, iterator _end) :
    z {_z}, begin {_begin}, end {_end} {};
  __host__ __device__ SeedZWithIteratorPair() {};

  PVTrack* get_array() const
  {
    std::vector<PVTrack> track_vec(begin, end);
    return track_vec.data();
  };

  uint get_size() const
  {
    std::vector<PVTrack> track_vec(begin, end);
    return track_vec.size();
  };
};

// Need a small extension to the track when fitting the
// vertex. Caching this information doesn't seem to help much
// though.
struct PVTrackInVertex : PVTrack {
  __host__ __device__ PVTrackInVertex(const PVTrack& trk) : PVTrack {trk}
  {
    // H matrix is symmetric and has four non-zero entries
    H_00 = 1.f;
    H_11 = 1.f;
    H_20 = -trk.tx.x;
    H_21 = -trk.tx.y;
    // HW: product of H and W matrices, symmetric with four non-zero entries
    HW_00 = W_00;
    HW_11 = W_11;
    HW_20 = H_20 * W_00;
    HW_21 = H_21 * W_11;
    // HWH: ROOT::Math::Similarity(H,W)
    HWH_00 = W_00;
    HWH_20 = H_20 * W_00;
    HWH_11 = W_11;
    HWH_21 = H_21 * W_11;
    HWH_22 = H_20 * H_20 * W_00 + H_21 * H_21 * W_11;
  }
  // TO DO: check whether this needs to be a double
  // changing it to float slightly (~ 1%) degrades the efficency and fake rate
  float H_00;
  float H_11;
  float H_20;
  float H_21;
  // HW: product of H and W matrices, symmetric with four non-zero entries
  float HW_00;
  float HW_11;
  float HW_20;
  float HW_21;
  float HWH_00;
  float HWH_20;
  float HWH_11;
  float HWH_21;
  float HWH_22;
  float weight = 1.f;
};

void findPVs(
  char* kalmanvelo_states,
  int* velo_atomics,
  uint* velo_track_hit_number,
  PV::Vertex* reconstructed_pvs,
  int* number_of_pvs,
  const uint number_of_events);
