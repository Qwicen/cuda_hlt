#pragma once

#include "../../main/include/Common.h"
#include <cstring>

constexpr unsigned int max_cluster_size = 196608;

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

static const unsigned int NModules = 52;
static const unsigned int NSensorsPerModule = 4;
static const unsigned int NSensors = NModules * NSensorsPerModule;
static const unsigned int NChipsPerSensor = 3;
static const unsigned int NRows = 256;
static const unsigned int NColumns = 256;
static const unsigned int NSensorColumns = NColumns * NChipsPerSensor;
static const unsigned int NPixelsPerSensor = NSensorColumns * NRows;
static const unsigned int ChipColumns = 256;

static const double Pitch = 0.055;

}

uint32_t get_channel_id(
  unsigned int sensor,
  unsigned int chip,
  unsigned int col,
  unsigned int row
);

uint32_t get_lhcb_id(uint32_t cid);

void clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
);

std::vector<uint32_t> cuda_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
);

std::vector<uint32_t> cuda_array_clustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned int>& event_offsets
);

void cache_sp_patterns(
  unsigned char* sp_patterns,
  unsigned char* sp_sizes,
  float* sp_fx,
  float* sp_fy
);
