#pragma once

#include <stdint.h>
#include <vector>
#include <ostream>

static constexpr uint32_t ut_number_of_sectors_per_board = 6;
static constexpr uint32_t ut_number_of_geometry_sectors = 1048;

struct UTBoards {
  uint32_t number_of_boards;
  uint32_t number_of_channels;
  uint32_t* stripsPerHybrids;
  uint32_t* stations;
  uint32_t* layers;
  uint32_t* detRegions;
  uint32_t* sectors;
  uint32_t* chanIDs;

  UTBoards(const std::vector<char> & ut_boards);
  
  __device__ __host__ UTBoards (
    const char* ut_boards
  );
};

struct UTGeometry {
  
  uint32_t number_of_sectors;
  uint32_t* firstStrip;
  float* pitch;
  float* dy;
  float* dp0diX;
  float* dp0diY;
  float* dp0diZ;
  float* p0X;
  float* p0Y;
  float* p0Z;
  float* cos;

  UTGeometry(const std::vector<char> & ut_geometry);
  
  __device__ __host__ UTGeometry (
    const char* ut_geometry
  );
};

struct  UTRawBank {
  uint32_t sourceID;
  uint32_t number_of_hits;
  uint16_t* data;

  __device__ __host__ UTRawBank (
    const uint32_t* ut_raw_bank
  );
};

struct  UTRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offsets;
  uint32_t* data;

  __device__ __host__ UTRawEvent (
    const uint32_t* ut_raw_event
  );

  __device__ __host__ UTRawBank getUTRawBank(
    const uint32_t index
  ) const;
};

struct UTHit {
  float cos;
  float yBegin;
  float yEnd;
  float zAtYEq0;
  float xAtYEq0;
  float weight;
  uint32_t highThreshold;
  uint32_t LHCbID;
  uint32_t planeCode;

  // int layer_offset[n_layers];
  // int n_hits_layers[n_layers];
  
  // float m_cos[max_numhits_per_event];
  // float m_yBegin[max_numhits_per_event];
  // float m_yEnd[max_numhits_per_event];
  // float m_zAtYEq0[max_numhits_per_event];
  // float m_xAtYEq0[max_numhits_per_event];
  // float m_weight[max_numhits_per_event];
  // int   m_highThreshold[max_numhits_per_event];
  // unsigned int m_LHCbID[max_numhits_per_event];
  // int m_planeCode[max_numhits_per_event];

  UTHit() = default;

  UTHit(float cos,
        float yBegin,
        float yEnd,
        float zAtYEq0,
        float xAtYEq0,
        float weight,
        uint32_t highThreshold,
        uint32_t LHCbID,
        uint32_t planeCode
        );

  #define cmpf(a, b) (fabs((a) - (b)) > 0.000065f)

  bool operator!=(const UTHit & h) const {
    if (cmpf(cos,        h.cos))          return true;
    if (cmpf(yBegin,     h.yBegin))       return true;
    if (cmpf(yEnd,       h.yEnd))         return true;
    if (cmpf(zAtYEq0,    h.zAtYEq0))      return true;
    if (cmpf(xAtYEq0,    h.xAtYEq0))      return true;
    if (cmpf(weight,     h.weight))       return true;
    if (highThreshold != h.highThreshold) return true;
    if (LHCbID        != h.LHCbID)        return true;
    
    return false;
  }

  bool operator==(const UTHit& h) const {
    return !(*this != h);
  }

  friend std::ostream& operator<<(std::ostream& stream, const UTHit& ut_hit) {
    stream << "UT hit {"
      << ut_hit.LHCbID << ", "
      << ut_hit.cos << ", "
      << ut_hit.yBegin << ", "
      << ut_hit.yEnd << ", "
      << ut_hit.zAtYEq0 << ", "
      << ut_hit.xAtYEq0 << ", "
      << ut_hit.weight << ", "
      << ut_hit.highThreshold << ", "
      << ut_hit.planeCode << "}";

    return stream;
  }
};

  /*      Copied from cuda/veloUT/common/include/VeloUTDefinitions.cuh    */
static constexpr uint ut_number_of_layers = 4;
static constexpr uint ut_max_number_of_hits_per_layer = 1024;
static constexpr uint ut_max_number_of_hits_per_event = ut_number_of_layers* ut_max_number_of_hits_per_layer;

/**
* @brief Offset and number of hits of each layer.
* 
*/
struct UTHitCount {
  uint32_t layer_offset [ut_number_of_layers];
  uint32_t n_hits_layers[ut_number_of_layers];
};

/* 
   SoA for hit variables
   The hits for every layer are written behind each other, the offsets 
   are stored for access;
   one Hits structure exists per event
*/
struct UTHits {
  float m_cos     [ut_max_number_of_hits_per_event];
  float m_yBegin  [ut_max_number_of_hits_per_event];
  float m_yEnd    [ut_max_number_of_hits_per_event];
  float m_zAtYEq0 [ut_max_number_of_hits_per_event];
  float m_xAtYEq0 [ut_max_number_of_hits_per_event];
  float m_weight  [ut_max_number_of_hits_per_event];
  
  uint32_t m_highThreshold  [ut_max_number_of_hits_per_event];
  uint32_t m_LHCbID         [ut_max_number_of_hits_per_event];
  uint32_t m_planeCode      [ut_max_number_of_hits_per_event];

  UTHit getHit(uint32_t index, uint32_t layer) const;
};
