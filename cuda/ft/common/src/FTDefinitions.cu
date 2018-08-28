#include "FTDefinitions.cuh"

/*                     ---Geometry format---
| name                         | type     | size               |
|------------------------------|----------|--------------------|
| number_of_stations           | uint32_t | 1                  |
| number_of_layers_per_station | uint32_t | 1                  |
| number_of_quarters_per_layer | uint32_t | 1                  |
| number_of_modules            | uint32_t | number_of_quarters |
| number_of_mats_per_module    | uint32_t | 1                  |
| number_of_tell40s            | uint32_t | 1                  |
| bank_first_channel           | uint32_t | number_of_tell40s  | //readout map as generated in FTDAQ/FTReadoutTool
| mirrorPoint_x                | float    | number_of_mats     |
| mirrorPoint_y                | float    | number_of_mats     |
| mirrorPoint_z                | float    | number_of_mats     |
| ddx_x                        | float    | number_of_mats     |
| ddx_y                        | float    | number_of_mats     |
| ddx_z                        | float    | number_of_mats     |
| uBegin                       | float    | number_of_mats     |
| halfChannelPitch             | float    | number_of_mats     |
| dieGap                       | float    | number_of_mats     |
| sipmPitch                    | float    | number_of_mats     |
| dxdy                         | float    | number_of_mats     |
| dzdy                         | float    | number_of_mats     |
| globaldy                     | float    | number_of_mats     |
*/

__device__ __host__ FTGeometry::FTGeometry(
  const char* geometry
) {
  const char* p = geometry;

  number_of_stations           = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers_per_station = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_quarters_per_layer = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers             = number_of_stations * number_of_layers_per_station;
  number_of_quarters           = number_of_quarters_per_layer * number_of_layers;
  number_of_modules            = (uint32_t*)p; p += number_of_quarters * sizeof(uint32_t);
  number_of_mats_per_module    = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_mats               = 0;
  for(size_t quarter = 0; quarter < number_of_quarters; quarter++)
    number_of_mats += number_of_modules[quarter] * number_of_mats_per_module;
  number_of_tell40s            = *((uint32_t*)p); p += sizeof(uint32_t);
  bank_first_channel           = (uint32_t*)p; p += number_of_tell40s * sizeof(uint32_t);
  mirrorPoint_x                = (float*)p; p += sizeof(float) * number_of_mats;
  mirrorPoint_y                = (float*)p; p += sizeof(float) * number_of_mats;
  mirrorPoint_z                = (float*)p; p += sizeof(float) * number_of_mats;
  ddx_x                        = (float*)p; p += sizeof(float) * number_of_mats;
  ddx_y                        = (float*)p; p += sizeof(float) * number_of_mats;
  ddx_z                        = (float*)p; p += sizeof(float) * number_of_mats;
  uBegin                       = (float*)p; p += sizeof(float) * number_of_mats;
  halfChannelPitch             = (float*)p; p += sizeof(float) * number_of_mats;
  dieGap                       = (float*)p; p += sizeof(float) * number_of_mats;
  sipmPitch                    = (float*)p; p += sizeof(float) * number_of_mats;
  dxdy                         = (float*)p; p += sizeof(float) * number_of_mats;
  dzdy                         = (float*)p; p += sizeof(float) * number_of_mats;
  globaldy                     = (float*)p; p += sizeof(float) * number_of_mats;

  size = p - geometry;
}

FTGeometry::FTGeometry(const std::vector<char>& geometry) : FTGeometry::FTGeometry(geometry.data()) {}

__device__ __host__ FTRawEvent::FTRawEvent(
  const char* event
) {
  const char* p = event;
  number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
  version = *((uint32_t*)p); p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p; p += number_of_raw_banks * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ FTRawBank::FTRawBank(const char* raw_bank, const char* end) {
  const char* p = raw_bank;
  sourceID = *((uint32_t*)p); p += sizeof(uint32_t);
  data = (uint16_t*) p;
  last = (uint16_t*) --end;
}

__device__ __host__ FTChannelID FTChannelID::operator+(const uint32_t& other){
  channelID += other;
  return *this;
}

__device__ __host__ uint8_t FTChannelID::channel() const {
  return (uint8_t)((channelID & channelMask) >> channelBits);
}

__device__ __host__ uint8_t FTChannelID::sipm() const {
  return (uint8_t)((channelID & sipmMask) >> sipmBits);
}

__device__ __host__ uint8_t FTChannelID::mat() const {
  return (uint8_t)((channelID & matMask) >> matBits);
}

__device__ __host__ uint8_t FTChannelID::module() const {
  return (uint8_t)((channelID & moduleMask) >> moduleBits);
}

__device__ __host__ uint8_t FTChannelID::quarter() const {
  return (uint8_t)((channelID & quarterMask) >> quarterBits);
}

__device__ __host__ uint8_t FTChannelID::layer() const {
  return (uint8_t)((channelID & layerMask) >> layerBits);
}

__device__ __host__ uint8_t FTChannelID::station() const{
  return (uint8_t)((channelID & stationMask) >> stationBits);
}
