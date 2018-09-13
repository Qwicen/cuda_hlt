#pragma once

#include <stdint.h>
#include <vector>
#include <iostream>

/**
 * @brief FT geometry description typecast.
 */
namespace FT {
struct FTGeometry {
  size_t size;
  uint32_t number_of_stations;
  uint32_t number_of_layers_per_station;
  uint32_t number_of_layers;
  uint32_t number_of_quarters_per_layer;
  uint32_t number_of_quarters;
  uint32_t* number_of_modules; //for each quarter
  uint32_t number_of_mats_per_module;
  uint32_t number_of_mats;
  uint32_t number_of_tell40s;
  uint32_t* bank_first_channel;
  float* mirrorPoint_x;
  float* mirrorPoint_y;
  float* mirrorPoint_z;
  float* ddx_x;
  float* ddx_y;
  float* ddx_z;
  float* uBegin;
  float* halfChannelPitch;
  float* dieGap;
  float* sipmPitch;
  float* dxdy;
  float* dzdy;
  float* globaldy;

  /**
   * @brief Typecast from std::vector.
   */
  FTGeometry(const std::vector<char>& geometry);

  /**
   * @brief Just typecast, no size check.
   */
  __device__ __host__ FTGeometry(
    const char* geometry
  );
};

struct FTRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  uint32_t version;
  char* payload;

  __device__ __host__ FTRawEvent(
    const char* event
  );
};

struct FTRawBank {
  uint32_t sourceID;
  uint16_t* data;
  uint16_t* last;

  __device__ __host__ FTRawBank(const char* raw_bank, const char* end);
};

namespace FTRawBankParams { //from FT/FTDAQ/src/FTRawBankParams.h
  enum shifts {
    linkShift     = 9,
    cellShift     = 2,
    fractionShift = 1,
    sizeShift     = 0,
  };

  static constexpr uint16_t nbClusMaximum   = 31;  // 5 bits
  static constexpr uint16_t nbClusFFMaximum = 10;  //
  static constexpr uint16_t fractionMaximum = 1;   // 1 bits allocted
  static constexpr uint16_t cellMaximum     = 127; // 0 to 127; coded on 7 bits
  static constexpr uint16_t sizeMaximum     = 1;   // 1 bits allocated

  enum BankProperties {
    NbBanks = 240,
    NbLinksPerBank = 24
  };

  static constexpr uint16_t clusterMaxWidth = 4;
}


struct FTChannelID {
  uint32_t channelID;
  __device__ __host__ uint8_t channel() const;
  __device__ __host__ uint8_t sipm() const;
  __device__ __host__ uint8_t mat() const;
  __device__ __host__ uint8_t module() const;
  __device__ __host__ uint8_t quarter() const;
  __device__ __host__ uint8_t layer() const;
  __device__ __host__ uint8_t station() const;
  __device__ __host__ FTChannelID operator+=(const uint32_t& other);
  __host__ std::string toString();
  //from FTChannelID.h (generated)
  enum channelIDMasks{channelMask       = 0x7fL,
                      sipmMask          = 0x180L,
                      matMask           = 0x600L,
                      moduleMask        = 0x3800L,
                      quarterMask       = 0xc000L,
                      layerMask         = 0x30000L,
                      stationMask       = 0xc0000L};
  enum channelIDBits{channelBits       = 0,
                     sipmBits          = 7,
                     matBits           = 9,
                     moduleBits        = 11,
                     quarterBits       = 14,
                     layerBits         = 16,
                     stationBits       = 18};
};

struct FTLiteCluster {
  FTChannelID channelID;
  uint8_t fraction;
  uint8_t pseudoSize;
};
}
