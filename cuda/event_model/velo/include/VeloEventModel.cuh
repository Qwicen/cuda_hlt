#pragma once

#include <stdint.h>
#include "cuda_runtime.h"
#include "VeloDefinitions.cuh"

namespace Velo {

  struct Module {
    uint hitStart;
    uint hitNums;
    float z;

    __device__ Module() {}
    __device__ Module(const uint _hitStart, const uint _hitNums, const float _z) :
      hitStart(_hitStart), hitNums(_hitNums), z(_z)
    {}
  };

  struct HitBase { // 3 * 4 = 16 B
    float x;
    float y;
    float z;

    __device__ HitBase() {}

    __device__ HitBase(const float _x, const float _y, const float _z) : x(_x), y(_y), z(_z) {}
  };

  struct Hit : public HitBase { // 4 * 4 = 16 B
    uint LHCbID;

    __device__ Hit() {}

    __device__ Hit(const float _x, const float _y, const float _z, const uint _LHCbID) :
      HitBase(_x, _y, _z), LHCbID(_LHCbID)
    {}
  };

  /**
   * @brief TrackletHits struct
   */
  struct TrackletHits { // 3 * 2 = 6 B
    unsigned short hits[3];

    __device__ TrackletHits() {}
    __device__ TrackletHits(const unsigned short h0, const unsigned short h1, const unsigned short h2)
    {
      hits[0] = h0;
      hits[1] = h1;
      hits[2] = h2;
    }
  };

  /* Structure containing indices to hits within hit array */
  struct TrackHits { // 2 + 26 * 2 = 54 B
    unsigned short hitsNum;
    unsigned short hits[Velo::Constants::max_track_size];

    __device__ TrackHits() {}

    __device__ TrackHits(
      const unsigned short _hitsNum,
      const unsigned short _h0,
      const unsigned short _h1,
      const unsigned short _h2) :
      hitsNum(_hitsNum)
    {
      hits[0] = _h0;
      hits[1] = _h1;
      hits[2] = _h2;
    }

    __device__ TrackHits(const TrackletHits& tracklet)
    {
      hitsNum = 3;
      hits[0] = tracklet.hits[0];
      hits[1] = tracklet.hits[1];
      hits[2] = tracklet.hits[2];
    }
  };

  /**
   * @brief Structure to save final track
   * Contains information needed later on in the HLT chain
   * and / or for truth matching.
   */
  struct Track { // 4 + 26 * 16 = 420 B
    bool backward;
    unsigned short hitsNum;
    Hit hits[Velo::Constants::max_track_size];

    __device__ Track() { hitsNum = 0; }

    __device__ void addHit(const Hit& _h)
    {
      hits[hitsNum] = _h;
      hitsNum++;
    }
  };

  /**
   * @brief Means square fit parameters
   *        required for Kalman fit (Velo)
   */
  struct TrackFitParameters {
    float tx, ty;
    bool backward;
  };

} // namespace Velo
