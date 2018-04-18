#pragma once

#include <stdint.h>
#include <vector>
#include <iostream>

namespace VeloClustering {
  extern __constant__ uint8_t sp_patterns [256];
  extern __constant__ uint8_t sp_sizes [256];
  extern __constant__ float sp_fx [512];
  extern __constant__ float sp_fy [512];

  static constexpr uint32_t mask_bottom          = 0xFFFEFFFF;
  static constexpr uint32_t mask_top             = 0xFFFF7FFF;
  static constexpr uint32_t mask_top_left        = 0x7FFF7FFF;
  static constexpr uint32_t mask_bottom_right    = 0xFFFEFFFE;
  static constexpr uint32_t mask_ltr_top_right   = 0x7FFF0000;
  static constexpr uint32_t mask_rtl_bottom_left = 0x0000FFFE;
  static constexpr uint32_t max_clustering_iterations = 12;
  static constexpr uint32_t max_candidates_event = 2000;
}

namespace LHCb {
  namespace VPChannelID {
    /// Offsets of bitfield channelID
    enum channelIDBits{rowBits    = 0,
                       colBits    = 8,
                       chipBits   = 16,
                       sensorBits = 18
                     };

    /// Bitmasks for bitfield channelID
    enum channelIDMasks{rowMask    = 0xffL,
                        colMask    = 0xff00L,
                        chipMask   = 0x30000L,
                        sensorMask = 0xffc0000L
                      };

    enum channelIDtype{ Velo=1,
                        TT,
                        IT,
                        OT,
                        Rich,
                        Calo,
                        Muon,
                        VP,
                        FT=10,
                        UT,
                        HC
                      };
  }

  /// Offsets of bitfield lhcbID
  enum lhcbIDBits{IDBits           = 0,
                  detectorTypeBits = 28};
}

namespace VP {
  static constexpr uint NModules = 52;
  static constexpr uint NSensorsPerModule = 4;
  static constexpr uint NSensors = NModules * NSensorsPerModule;
  static constexpr uint NChipsPerSensor = 3;
  static constexpr uint NRows = 256;
  static constexpr uint NColumns = 256;
  static constexpr uint NSensorColumns = NColumns * NChipsPerSensor;
  static constexpr uint NPixelsPerSensor = NSensorColumns * NRows;
  static constexpr uint ChipColumns = 256;
  static constexpr double Pitch = 0.055;
}

struct VeloRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  char* payload;

  __device__ __host__ VeloRawEvent(
    const char* event,
    uint = 0
  );
};

struct VeloRawBank {
  uint32_t sensor_index;
  uint32_t sp_count;
  uint32_t* sp_word;

  __device__ __host__ VeloRawBank(
    const char* raw_bank,
    uint = 0
  );
};

/**
 * @brief Velo geometry description typecast.
 */
struct VeloGeometry {
  size_t size;
  uint32_t number_of_sensors;
  uint32_t number_of_sensor_columns;
  uint32_t number_of_sensors_per_module;
  uint32_t chip_columns;
  float pixel_size;
  double* local_x;
  double* x_pitch;
  float* ltg;

  /**
   * @brief Typecast from std::vector.
   */
  VeloGeometry(const std::vector<char>& geometry);

  /**
   * @brief Just typecast, no size check.
   */
  __device__ __host__ VeloGeometry(
    const char* geometry
  );
};

__device__ __host__ uint32_t get_channel_id(
  uint sensor,
  uint chip,
  uint col,
  uint row
);

__device__ __host__ uint32_t get_lhcb_id(uint32_t cid);
