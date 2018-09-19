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
| max_uniqueMat                | uint32_t | 1                  |
| mirrorPointX                 | float    | max_uniqueMat      |
| mirrorPointY                 | float    | max_uniqueMat      |
| mirrorPointZ                 | float    | max_uniqueMat      |
| ddxX                         | float    | max_uniqueMat      |
| ddxY                         | float    | max_uniqueMat      |
| ddxZ                         | float    | max_uniqueMat      |
| uBegin                       | float    | max_uniqueMat      |
| halfChannelPitch             | float    | max_uniqueMat      |
| dieGap                       | float    | max_uniqueMat      |
| sipmPitch                    | float    | max_uniqueMat      |
| dxdy                         | float    | max_uniqueMat      |
| dzdy                         | float    | max_uniqueMat      |
| globaldy                     | float    | max_uniqueMat      |
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
  max_uniqueMat                = *((uint32_t*)p); p += sizeof(uint32_t);
  mirrorPointX                 = (float*)p; p += sizeof(float) * max_uniqueMat;
  mirrorPointY                 = (float*)p; p += sizeof(float) * max_uniqueMat;
  mirrorPointZ                 = (float*)p; p += sizeof(float) * max_uniqueMat;
  ddxX                         = (float*)p; p += sizeof(float) * max_uniqueMat;
  ddxY                         = (float*)p; p += sizeof(float) * max_uniqueMat;
  ddxZ                         = (float*)p; p += sizeof(float) * max_uniqueMat;
  uBegin                       = (float*)p; p += sizeof(float) * max_uniqueMat;
  halfChannelPitch             = (float*)p; p += sizeof(float) * max_uniqueMat;
  dieGap                       = (float*)p; p += sizeof(float) * max_uniqueMat;
  sipmPitch                    = (float*)p; p += sizeof(float) * max_uniqueMat;
  dxdy                         = (float*)p; p += sizeof(float) * max_uniqueMat;
  dzdy                         = (float*)p; p += sizeof(float) * max_uniqueMat;
  globaldy                     = (float*)p; p += sizeof(float) * max_uniqueMat;

  size = p - geometry;
}

FTGeometry::FTGeometry(const std::vector<char>& geometry) : FTGeometry::FTGeometry(geometry.data()) {}

__device__ __host__ FTRawEvent::FTRawEvent(const char* event) {
  const char* p = event;
  number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p; p += (number_of_raw_banks + 1) * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ FTRawBank FTRawEvent::getFTRawBank(const uint32_t index) const {
  FTRawBank bank(payload + raw_bank_offset[index], payload + raw_bank_offset[index + 1]);
  return bank;
}

__device__ __host__ FTRawBank::FTRawBank(const char* raw_bank, const char* end) {
  const char* p = raw_bank;
  sourceID = *((uint32_t*)p); p += sizeof(uint32_t);
  data = (uint16_t*) p;
  last = (uint16_t*) end;
}

__device__ __host__ FTChannelID::FTChannelID(const uint32_t channelID) : channelID(channelID) {};

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

void FTHits::typecast_unsorted(char* base, uint32_t total_number_of_hits) {
  x0    =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  z0    =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  w     =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  dxdy  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  dzdy  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  yMin  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  yMax  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  werrX =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  coord =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  LHCbID =    reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  planeCode = reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  hitZone =   reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  info =      reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  used =      reinterpret_cast<bool*>(base); base += sizeof(bool) * total_number_of_hits;
  temp  =     reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
}

void FTHits::typecast_sorted(char* base, uint32_t total_number_of_hits) {
  temp  =     reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  x0    =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  z0    =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  w     =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  dxdy  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  dzdy  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  yMin  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  yMax  =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  werrX =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  coord =     reinterpret_cast<float*>(base); base += sizeof(float) * total_number_of_hits;
  LHCbID =    reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  planeCode = reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  hitZone =   reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  info =      reinterpret_cast<uint32_t*>(base); base += sizeof(uint32_t) * total_number_of_hits;
  used =      reinterpret_cast<bool*>(base); base += sizeof(bool) * total_number_of_hits;
}

FTHit FTHits::getHit(uint32_t index) const {
  return {x0[index], z0[index], w[index], dxdy[index], dzdy[index], yMin[index],
          yMax[index], werrX[index], coord[index], LHCbID[index], planeCode[index],
          hitZone[index], info[index], used[index]};
}
};
