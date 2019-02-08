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

  __device__ __host__ SciFiGeometry::SciFiGeometry(const char* geometry)
  {
    const char* p = geometry;

    number_of_stations = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_layers_per_station = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_layers = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_quarters_per_layer = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_quarters = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_modules = (uint32_t*) p;
    p += number_of_quarters * sizeof(uint32_t);
    number_of_mats_per_module = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_mats = *((uint32_t*) p);
    p += sizeof(uint32_t);
    number_of_tell40s = *((uint32_t*) p);
    p += sizeof(uint32_t);
    bank_first_channel = (uint32_t*) p;
    p += number_of_tell40s * sizeof(uint32_t);
    max_uniqueMat = *((uint32_t*) p);
    p += sizeof(uint32_t);
    mirrorPointX = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    mirrorPointY = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    mirrorPointZ = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    ddxX = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    ddxY = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    ddxZ = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    uBegin = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    halfChannelPitch = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    dieGap = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    sipmPitch = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    dxdy = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    dzdy = (float*) p;
    p += sizeof(float) * max_uniqueMat;
    globaldy = (float*) p;
    p += sizeof(float) * max_uniqueMat;

    size = p - geometry;
  }

  SciFiGeometry::SciFiGeometry(const std::vector<char>& geometry) : SciFiGeometry::SciFiGeometry(geometry.data()) {}

  __device__ __host__ SciFiChannelID::SciFiChannelID(const uint32_t channelID) : channelID(channelID) {};

  __host__ std::string SciFiChannelID::toString()
  {
    std::ostringstream s;
    s << "{ SciFiChannelID : "
      << " channel =" << std::to_string(channel()) << " sipm =" << std::to_string(sipm())
      << " mat =" << std::to_string(mat()) << " module=" << std::to_string(module())
      << " quarter=" << std::to_string(quarter()) << " layer=" << std::to_string(layer())
      << " station=" << std::to_string(station()) << " }";
    return s.str();
  }

  __device__ __host__ uint32_t SciFiChannelID::channel() const
  {
    return (uint32_t)((channelID & channelMask) >> channelBits);
  }

  __device__ __host__ uint32_t SciFiChannelID::sipm() const { return ((channelID & sipmMask) >> sipmBits); }

  __device__ __host__ uint32_t SciFiChannelID::mat() const { return ((channelID & matMask) >> matBits); }

  __device__ __host__ uint32_t SciFiChannelID::module() const { return ((channelID & moduleMask) >> moduleBits); }

  __device__ __host__ uint SciFiChannelID::correctedModule() const
  {
    // Returns local module ID in ascending x order.
    // There may be a faster way to do this.
    uint uQuarter = uniqueQuarter() - 16;
    uint module_count = uQuarter >= 32 ? 6 : 5;
    uint q = uQuarter % 4;
    if (q == 0 || q == 2) return module_count - 1 - module();
    if (q == 1 || q == 3) return module();
    return 0;
  };

  __device__ __host__ uint32_t SciFiChannelID::quarter() const { return ((channelID & quarterMask) >> quarterBits); }

  __device__ __host__ uint32_t SciFiChannelID::layer() const { return ((channelID & layerMask) >> layerBits); }

  __device__ __host__ uint32_t SciFiChannelID::station() const { return ((channelID & stationMask) >> stationBits); }

  __device__ __host__ uint32_t SciFiChannelID::uniqueLayer() const
  {
    return ((channelID & uniqueLayerMask) >> layerBits);
  }

  __device__ __host__ uint32_t SciFiChannelID::uniqueMat() const { return ((channelID & uniqueMatMask) >> matBits); }

  __device__ __host__ uint32_t SciFiChannelID::correctedUniqueMat() const
  {
    // Returns global mat ID in ascending x order without any gaps.
    // Geometry dependent. No idea how to not hardcode this.
    uint32_t quarter = uniqueQuarter() - 16;
    return (quarter < 32 ? quarter : 32) * 5 * 4 + (quarter >= 32 ? quarter - 32 : 0) * 6 * 4 + 4 * correctedModule() +
           (reversedZone() ? 3 - mat() : mat());
  }

  __device__ __host__ uint32_t SciFiChannelID::uniqueModule() const
  {
    return ((channelID & uniqueModuleMask) >> moduleBits);
  }

  __device__ __host__ uint32_t SciFiChannelID::uniqueQuarter() const
  {
    return ((channelID & uniqueQuarterMask) >> quarterBits);
  }

  __device__ __host__ uint32_t SciFiChannelID::die() const { return ((channelID & 0x40) >> 6); }

  __device__ __host__ bool SciFiChannelID::isBottom() const { return (quarter() == 0 || quarter() == 1); }

  __device__ __host__ bool SciFiChannelID::reversedZone() const
  {
    uint zone = ((uniqueQuarter() - 16) >> 1) % 4;
    return zone == 1 || zone == 2;
  };

  __device__ uint32_t channelInBank(uint32_t c) { return (c >> SciFiRawBankParams::cellShift); }

  __device__ uint16_t getLinkInBank(uint16_t c) { return (c >> SciFiRawBankParams::linkShift); }

  __device__ int cell(uint16_t c) { return (c >> SciFiRawBankParams::cellShift) & SciFiRawBankParams::cellMaximum; }

  __device__ int fraction(uint16_t c)
  {
    return (c >> SciFiRawBankParams::fractionShift) & SciFiRawBankParams::fractionMaximum;
  }

  __device__ bool cSize(uint16_t c) { return (c >> SciFiRawBankParams::sizeShift) & SciFiRawBankParams::sizeMaximum; }

}; // namespace SciFi
