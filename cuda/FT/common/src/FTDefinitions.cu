#include "FTDefinitions.cuh"
#include <sstream>

/*                     ---Geometry format---
| name                         | type     | size               |
|------------------------------|----------|--------------------|
| number_of_stations           | uint32_t | 1                  |
| number_of_layers_per_station | uint32_t | 1                  |
| number_of_layers             | uint32_t | 1                  |
| number_of_quarters_per_layer | uint32_t | 1                  |
| number_of_quarters           | uint32_t | 1                  |
| number_of_modules            | uint32_t | number_of_quarters |
| number_of_mats_per_module    | uint32_t | 1                  |
| number_of_mats               | uint32_t | 1                  |
| number_of_tell40s            | uint32_t | 1                  |
| bank_first_channel           | uint32_t | number_of_tell40s  |
| mirrorPointX                 | float    | number_of_mats     |
| mirrorPointY                 | float    | number_of_mats     |
| mirrorPointZ                 | float    | number_of_mats     |
| ddxX                         | float    | number_of_mats     |
| ddxY                         | float    | number_of_mats     |
| ddxZ                         | float    | number_of_mats     |
| uBegin                       | float    | number_of_mats     |
| halfChannelPitch             | float    | number_of_mats     |
| dieGap                       | float    | number_of_mats     |
| sipmPitch                    | float    | number_of_mats     |
| dxdy                         | float    | number_of_mats     |
| dzdy                         | float    | number_of_mats     |
| globaldy                     | float    | number_of_mats     |
*/

namespace FT {

__device__ __host__ FTGeometry::FTGeometry(
  const char* geometry
) {
  const char* p = geometry;

  number_of_stations           = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers_per_station = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers             = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_quarters_per_layer = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_quarters           = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_modules            = (uint32_t*)p; p += number_of_quarters * sizeof(uint32_t);
  number_of_mats_per_module    = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_mats               = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_tell40s            = *((uint32_t*)p); p += sizeof(uint32_t);
  bank_first_channel           = (uint32_t*)p; p += number_of_tell40s * sizeof(uint32_t);
  mirrorPointX                 = (float*)p; p += sizeof(float) * number_of_mats;
  mirrorPointY                 = (float*)p; p += sizeof(float) * number_of_mats;
  mirrorPointZ                 = (float*)p; p += sizeof(float) * number_of_mats;
  ddxX                         = (float*)p; p += sizeof(float) * number_of_mats;
  ddxY                         = (float*)p; p += sizeof(float) * number_of_mats;
  ddxZ                         = (float*)p; p += sizeof(float) * number_of_mats;
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
  //version = *((uint32_t*)p); p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p; p += (number_of_raw_banks + 1) * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ FTRawBank::FTRawBank(const char* raw_bank, const char* end) {
  const char* p = raw_bank;
  sourceID = *((uint32_t*)p); p += sizeof(uint32_t);
  data = (uint16_t*) p;
  last = (uint16_t*) end;
}

__device__ __host__ FTChannelID::FTChannelID(const uint32_t channelID) : channelID(channelID) {};

__device__ __host__ FTChannelID FTChannelID::operator+=(const uint32_t& other){
  channelID += other;
  return *this;
}

__host__ std::string FTChannelID::toString()
{
  std::ostringstream s;
  s << "{ FTChannelID : "
    << " channel =" << std::to_string(channel())
    << " sipm ="    << std::to_string(sipm())
    << " mat ="     << std::to_string(mat())
    << " module="   << std::to_string(module())
    << " quarter="  << std::to_string(quarter())
    << " layer="    << std::to_string(layer())
    << " station="  << std::to_string(station())
    << " }";
  return s.str();
}

__device__ __host__ uint32_t FTChannelID::channel() const {
  return (uint32_t)((channelID & channelMask) >> channelBits);
}

__device__ __host__ uint32_t FTChannelID::sipm() const {
  return ((channelID & sipmMask) >> sipmBits);
}

__device__ __host__ uint32_t FTChannelID::mat() const {
  return ((channelID & matMask) >> matBits);
}

__device__ __host__ uint32_t FTChannelID::module() const {
  return ((channelID & moduleMask) >> moduleBits);
}

__device__ __host__ uint32_t FTChannelID::quarter() const {
  return ((channelID & quarterMask) >> quarterBits);
}

__device__ __host__ uint32_t FTChannelID::layer() const {
  return ((channelID & layerMask) >> layerBits);
}
__device__ __host__ uint32_t FTChannelID::station() const {
  return ((channelID & stationMask) >> stationBits);
}
__device__ __host__ uint32_t FTChannelID::uniqueLayer() const {
  return ((channelID & uniqueLayerMask) >> layerBits);
}
__device__ __host__ uint32_t FTChannelID::uniqueMat() const {
  return ((channelID & uniqueMatMask) >> matBits);
}
__device__ __host__ uint32_t FTChannelID::uniqueModule() const {
  return ((channelID & uniqueModuleMask) >> moduleBits);
}
__device__ __host__ uint32_t FTChannelID::uniqueQuarter() const {
  return ((channelID & uniqueQuarterMask) >> quarterBits);
}
__device__ __host__ uint32_t FTChannelID::die() const {
  return ((channelID & 0x40) >> 6);
}

__device__ __host__ bool FTChannelID::isBottom() const {
 return (quarter() == 0 || quarter() == 1);
}

void FTHitCount::typecast_before_prefix_sum(
  uint* base_pointer,
  const uint event_number
) {
  n_hits_layers = base_pointer + event_number * FT::number_of_zones;
}

void FTHitCount::typecast_after_prefix_sum(
  uint* base_pointer,
  const uint event_number,
  const uint number_of_events
) {
  layer_offsets = base_pointer + event_number *  FT::number_of_zones;
  n_hits_layers = base_pointer + number_of_events * FT::number_of_zones + 1 + event_number * FT::number_of_zones;
}


};
