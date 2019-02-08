#pragma once

#include <cstdint>
#include <ostream>
#include "UTDefinitions.cuh"

namespace UT {

  // Hit base containing just the geometrical information about the hit.
  struct Hit {
    float yBegin;
    float yEnd;
    float zAtYEq0;
    float xAtYEq0;
    float weight;
    uint32_t LHCbID;
    uint8_t plane_code;

    __device__ Hit() {}

    __device__ Hit(
      const float _yBegin,
      const float _yEnd,
      const float _zAtYEq0,
      const float _xAtYEq0,
      const float _weight,
      const uint32_t _LHCbID,
      const uint8_t _plane_code) :
      yBegin(_yBegin),
      yEnd(_yEnd), zAtYEq0(_zAtYEq0), xAtYEq0(_xAtYEq0), weight(_weight), LHCbID(_LHCbID), plane_code(_plane_code)
    {}

#define cmpf(a, b) (fabs((a) - (b)) > 0.000065f)

    bool operator!=(const Hit& h) const
    {
      if (cmpf(yBegin, h.yBegin)) return true;
      if (cmpf(yEnd, h.yEnd)) return true;
      if (cmpf(zAtYEq0, h.zAtYEq0)) return true;
      if (cmpf(xAtYEq0, h.xAtYEq0)) return true;
      if (cmpf(weight, h.weight)) return true;
      if (LHCbID != h.LHCbID) return true;
      if (plane_code != h.plane_code) return true;
      return false;
    }

    bool operator==(const Hit& h) const { return !(*this != h); }

    friend std::ostream& operator<<(std::ostream& stream, const Hit& ut_hit)
    {
      stream << "UT hit {" << ut_hit.LHCbID << ", " << ut_hit.yBegin << ", " << ut_hit.yEnd << ", " << ut_hit.zAtYEq0
             << ", " << ut_hit.xAtYEq0 << ", " << ut_hit.weight << ut_hit.plane_code << "}";

      return stream;
    }
  };

  struct TrackHits {
    float qop;
    short hits[UT::Constants::max_track_size];
    unsigned short hits_num = 0;
    unsigned short velo_track_index;
  };

  /**
   * @brief Offset and number of hits of each layer.
   */
  struct HitOffsets {
    const uint* m_unique_x_sector_layer_offsets;
    const uint* m_ut_hit_offsets;
    const uint m_number_of_unique_x_sectors;

    __device__ __host__ HitOffsets(
      const uint* base_pointer,
      const uint event_number,
      const uint number_of_unique_x_sectors,
      const uint* unique_x_sector_layer_offsets) :
      m_unique_x_sector_layer_offsets(unique_x_sector_layer_offsets),
      m_ut_hit_offsets(base_pointer + event_number * number_of_unique_x_sectors),
      m_number_of_unique_x_sectors(number_of_unique_x_sectors)
    {}

    __device__ __host__ uint sector_group_offset(const uint sector_group) const
    {
      assert(sector_group <= m_number_of_unique_x_sectors);
      return m_ut_hit_offsets[sector_group];
    }

    __device__ __host__ uint sector_group_number_of_hits(const uint sector_group) const
    {
      assert(sector_group < m_number_of_unique_x_sectors);
      return m_ut_hit_offsets[sector_group + 1] - m_ut_hit_offsets[sector_group];
    }

    __device__ __host__ uint layer_offset(const uint layer_number) const
    {
      assert(layer_number < 4);
      return m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number]];
    }

    __device__ __host__ uint layer_number_of_hits(const uint layer_number) const
    {
      assert(layer_number < 4);
      return m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number + 1]] -
             m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number]];
    }

    __device__ __host__ uint event_offset() const { return m_ut_hit_offsets[0]; }

    __device__ __host__ uint event_number_of_hits() const
    {
      return m_ut_hit_offsets[m_number_of_unique_x_sectors] - m_ut_hit_offsets[0];
    }
  };

  /*
     SoA for hit variables
     The hits for every layer are written behind each other, the offsets
     are stored for access;
     one Hits structure exists per event
  */
  struct Hits {
    constexpr static uint number_of_arrays = 7;
    float* yBegin;
    float* yEnd;
    float* zAtYEq0;
    float* xAtYEq0;
    float* weight;
    uint32_t* LHCbID;
    uint32_t* raw_bank_index;

    /**
     * @brief Populates the UTHits object pointers to an array of data
     *        pointed by base_pointer.
     */
    __host__ __device__ Hits(uint32_t* base_pointer, uint32_t total_number_of_hits)
    {
      raw_bank_index = base_pointer;
      yBegin = reinterpret_cast<float*>(base_pointer + total_number_of_hits);
      yEnd = reinterpret_cast<float*>(base_pointer + 2 * total_number_of_hits);
      zAtYEq0 = reinterpret_cast<float*>(base_pointer + 3 * total_number_of_hits);
      xAtYEq0 = reinterpret_cast<float*>(base_pointer + 4 * total_number_of_hits);
      weight = reinterpret_cast<float*>(base_pointer + 5 * total_number_of_hits);
      LHCbID = base_pointer + 6 * total_number_of_hits;
    }

    /**
     * @brief Gets a hit in the UT::Hit format from the global hit index.
     */
    Hit getHit(uint32_t index) const
    {
      return {yBegin[index], yEnd[index], zAtYEq0[index], xAtYEq0[index], weight[index], LHCbID[index], 0};
    }

    __host__ __device__ inline bool isYCompatible(const int i_hit, const float y, const float tol) const
    {
      return yMin(i_hit) - tol <= y && y <= yMax(i_hit) + tol;
    }
    __host__ __device__ inline bool isNotYCompatible(const int i_hit, const float y, const float tol) const
    {
      return yMin(i_hit) - tol > y || y > yMax(i_hit) + tol;
    }
    __host__ __device__ inline float cosT(const int i_hit, const float dxDy) const
    {
      return (std::fabs(xAtYEq0[i_hit]) < 1.0E-9) ? 1. / std::sqrt(1 + dxDy * dxDy) : std::cos(dxDy);
    }
    __host__ __device__ inline float sinT(const int i_hit, const float dxDy) const
    {
      return tanT(i_hit, dxDy) * cosT(i_hit, dxDy);
    }
    __host__ __device__ inline float tanT(const int i_hit, const float dxDy) const { return -1 * dxDy; }
    __host__ __device__ inline float xAt(const int i_hit, const float globalY, const float dxDy) const
    {
      return xAtYEq0[i_hit] + globalY * dxDy;
    }
    __host__ __device__ inline float yMax(const int i_hit) const { return std::max(yBegin[i_hit], yEnd[i_hit]); }
    __host__ __device__ inline float yMid(const int i_hit) const { return 0.5 * (yBegin[i_hit] + yEnd[i_hit]); }
    __host__ __device__ inline float yMin(const int i_hit) const { return std::min(yBegin[i_hit], yEnd[i_hit]); }
  };
} // namespace UT
