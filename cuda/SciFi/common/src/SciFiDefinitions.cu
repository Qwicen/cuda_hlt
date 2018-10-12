#include "SciFiDefinitions.cuh"
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

namespace SciFi {

__device__ __host__ SciFiGeometry::SciFiGeometry(
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

SciFiGeometry::SciFiGeometry(const std::vector<char>& geometry) : SciFiGeometry::SciFiGeometry(geometry.data()) {}

__device__ __host__ SciFiRawEvent::SciFiRawEvent(const char* event) {
  const char* p = event;
  number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
  raw_bank_offset = (uint32_t*) p; p += (number_of_raw_banks + 1) * sizeof(uint32_t);
  payload = (char*) p;
}

__device__ __host__ SciFiRawBank SciFiRawEvent::getSciFiRawBank(const uint32_t index) const {
  SciFiRawBank bank(payload + raw_bank_offset[index], payload + raw_bank_offset[index + 1]);
  return bank;
}

__device__ __host__ SciFiRawBank::SciFiRawBank(const char* raw_bank, const char* end) {
  const char* p = raw_bank;
  sourceID = *((uint32_t*)p); p += sizeof(uint32_t);
  data = (uint16_t*) p;
  last = (uint16_t*) end;
}

__device__ __host__ SciFiChannelID::SciFiChannelID(const uint32_t channelID) : channelID(channelID) {};

__host__ std::string SciFiChannelID::toString()
{
  std::ostringstream s;
  s << "{ SciFiChannelID : "
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

__device__ __host__ uint32_t SciFiChannelID::channel() const {
  return (uint32_t)((channelID & channelMask) >> channelBits);
}

__device__ __host__ uint32_t SciFiChannelID::sipm() const {
  return ((channelID & sipmMask) >> sipmBits);
}

__device__ __host__ uint32_t SciFiChannelID::mat() const {
  return ((channelID & matMask) >> matBits);
}

__device__ __host__ uint32_t SciFiChannelID::module() const {
  return ((channelID & moduleMask) >> moduleBits);
}

__device__ __host__ uint32_t SciFiChannelID::quarter() const {
  return ((channelID & quarterMask) >> quarterBits);
}

__device__ __host__ uint32_t SciFiChannelID::layer() const {
  return ((channelID & layerMask) >> layerBits);
}
__device__ __host__ uint32_t SciFiChannelID::station() const {
  return ((channelID & stationMask) >> stationBits);
}
__device__ __host__ uint32_t SciFiChannelID::uniqueLayer() const {
  return ((channelID & uniqueLayerMask) >> layerBits);
}
__device__ __host__ uint32_t SciFiChannelID::uniqueMat() const {
  return ((channelID & uniqueMatMask) >> matBits);
}
__device__ __host__ uint32_t SciFiChannelID::uniqueModule() const {
  return ((channelID & uniqueModuleMask) >> moduleBits);
}
__device__ __host__ uint32_t SciFiChannelID::uniqueQuarter() const {
  return ((channelID & uniqueQuarterMask) >> quarterBits);
}
__device__ __host__ uint32_t SciFiChannelID::die() const {
  return ((channelID & 0x40) >> 6);
}

__device__ __host__ bool SciFiChannelID::isBottom() const {
 return (quarter() == 0 || quarter() == 1);
}

__device__ __host__
void SciFiHitCount::typecast_before_prefix_sum(
  uint* base_pointer,
  const uint event_number
) {
  n_hits_layers = base_pointer + event_number * SciFi::Constants::n_zones;
}

__device__ __host__
void SciFiHitCount::typecast_after_prefix_sum(
  uint* base_pointer,
  const uint event_number,
  const uint number_of_events
) {
  layer_offsets = base_pointer + event_number *  SciFi::Constants::n_zones;
  n_hits_layers = base_pointer + number_of_events * SciFi::Constants::n_zones + 1 + event_number * SciFi::Constants::n_zones;
}

void SciFiHits::typecast_unsorted(uint32_t* base, uint32_t total_number_of_hits) {
  x0    =     reinterpret_cast<float*>(base);
  z0    =     reinterpret_cast<float*>(base + total_number_of_hits);
  w     =     reinterpret_cast<float*>(base + 2*total_number_of_hits); 
  dxdy  =     reinterpret_cast<float*>(base + 3*total_number_of_hits); 
  dzdy  =     reinterpret_cast<float*>(base + 4*total_number_of_hits); 
  yMin  =     reinterpret_cast<float*>(base + 5*total_number_of_hits); 
  yMax  =     reinterpret_cast<float*>(base + 6*total_number_of_hits); 
  LHCbID =    base + 7*total_number_of_hits; 
  planeCode = base + 8*total_number_of_hits; 
  hitZone =   base + 9*total_number_of_hits; 
  temp  =     base + 10*total_number_of_hits; 
}

void SciFiHits::typecast_sorted(uint32_t* base, uint32_t total_number_of_hits) {
  temp  =     base; 
  x0    =     reinterpret_cast<float*>(base + total_number_of_hits); 
  z0    =     reinterpret_cast<float*>(base + 2*total_number_of_hits); 
  w     =     reinterpret_cast<float*>(base + 3*total_number_of_hits); 
  dxdy  =     reinterpret_cast<float*>(base + 4*total_number_of_hits); 
  dzdy  =     reinterpret_cast<float*>(base + 5*total_number_of_hits);
  yMin  =     reinterpret_cast<float*>(base + 6*total_number_of_hits); 
  yMax  =     reinterpret_cast<float*>(base + 7*total_number_of_hits); 
  LHCbID =    base + 8*total_number_of_hits; 
  planeCode = base + 9*total_number_of_hits; 
  hitZone =   base + 10*total_number_of_hits; 
}

SciFiHit SciFiHits::getHit(uint32_t index) const {
  return {x0[index], z0[index], w[index], dxdy[index], dzdy[index], yMin[index],
          yMax[index], LHCbID[index], planeCode[index], hitZone[index]};
}

};
