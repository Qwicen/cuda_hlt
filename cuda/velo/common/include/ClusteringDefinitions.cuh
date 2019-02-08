#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <vector>
#include "Logger.h"
#include "VeloDefinitions.cuh"

namespace VeloClustering {
  // Adjusted to minbias events. In the future, it should
  // be adjusted on the go.
  static constexpr uint32_t max_candidates_event = 3000;

  static constexpr uint32_t mask_bottom = 0xFFFEFFFF;
  static constexpr uint32_t mask_top = 0xFFFF7FFF;
  static constexpr uint32_t mask_top_left = 0x7FFF7FFF;
  static constexpr uint32_t mask_bottom_right = 0xFFFEFFFE;
  static constexpr uint32_t mask_ltr_top_right = 0x7FFF0000;
  static constexpr uint32_t mask_rtl_bottom_left = 0x0000FFFE;
  static constexpr uint32_t max_clustering_iterations = 12;
  static constexpr uint32_t lookup_table_size = 9;
} // namespace VeloClustering

namespace LHCb {
  namespace VPChannelID {
    /// Offsets of bitfield channelID
    enum channelIDBits { rowBits = 0, colBits = 8, chipBits = 16, sensorBits = 18 };

    /// Bitmasks for bitfield channelID
    enum channelIDMasks { rowMask = 0xffL, colMask = 0xff00L, chipMask = 0x30000L, sensorMask = 0xffc0000L };

    enum channelIDtype { Velo = 1, TT, IT, OT, Rich, Calo, Muon, VP, FT = 10, UT, HC };
  } // namespace VPChannelID

  /// Offsets of bitfield lhcbID
  enum lhcbIDBits { IDBits = 0, detectorTypeBits = 28 };
} // namespace LHCb

namespace VP {
  static constexpr uint NModules = Velo::Constants::n_modules;
  static constexpr uint NSensorsPerModule = 4;
  static constexpr uint NSensors = NModules * NSensorsPerModule;
  static constexpr uint NChipsPerSensor = 3;
  static constexpr uint NRows = 256;
  static constexpr uint NColumns = 256;
  static constexpr uint NSensorColumns = NColumns * NChipsPerSensor;
  static constexpr uint NPixelsPerSensor = NSensorColumns * NRows;
  static constexpr uint ChipColumns = 256;
  static constexpr double Pitch = 0.055;
} // namespace VP

struct VeloRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  char* payload;

  __device__ __host__ VeloRawEvent(const char* event);
};

struct VeloRawBank {
  uint32_t sensor_index;
  uint32_t sp_count;
  uint32_t* sp_word;

  __device__ __host__ VeloRawBank(const char* raw_bank);
};

/**
 * @brief Velo geometry description typecast.
 */
struct VeloGeometry {
  size_t size;
  float* local_x;
  float* x_pitch;
  float* ltg;

  /**
   * @brief Typecast from std::vector.
   */
  VeloGeometry(const std::vector<char>& geometry);

  /**
   * @brief Just typecast, no size check.
   */
  __device__ __host__ VeloGeometry(const char* geometry);
};

__device__ __host__ uint32_t get_channel_id(uint sensor, uint chip, uint col, uint row);

__device__ __host__ int32_t get_lhcb_id(int32_t cid);
