#include "FTDefinitions.cuh"

/*
                   ---Geometry format:---
| name                         | type     | size            |
|------------------------------|----------|-----------------|
| number_of_stations           | uint32_t | 1               |
| number_of_layers_per_station | uint32_t | 1               |
| number_of_modules            | uint32_t | number_of_layers|
| dxdy                         | float    | number_of_layers|
| dzdy                         | float    | number_of_layers|
| globaldy                     | float    | number_of_layers|
| endpoint_x                   | float    | number_of_layers|
| endpoint_y                   | float    | number_of_layers|
| endpoint_z                   | float    | number_of_layers|
*/

FTGeometry::FTGeometry(const std::vector<char>& geometry) {
  const char* p = geometry.data();

  number_of_stations           = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers_per_station = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers             = number_of_stations * number_of_layers_per_station;
  number_of_modules            = (uint32_t*)p; p += number_of_layers * sizeof(uint32_t);
  dxdy                         = (float*)p; p += sizeof(float) * number_of_layers;
  dzdy                         = (float*)p; p += sizeof(float) * number_of_layers;
  globaldy                     = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_x                   = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_y                   = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_z                   = (float*)p; p += sizeof(float) * number_of_layers;

  size = p - geometry.data();

  if (size != geometry.size()) {
    std::cout << "Size mismatch for geometry" << std::endl;
  }
}

__device__ __host__ FTGeometry::FTGeometry(
  const char* geometry
) {
  const char* p = geometry;

  number_of_stations           = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers_per_station = *((uint32_t*)p); p += sizeof(uint32_t);
  number_of_layers             = number_of_stations * number_of_layers_per_station;
  number_of_modules            = (uint32_t*)p; p += number_of_layers * sizeof(uint32_t);
  dxdy                         = (float*)p; p += sizeof(float) * number_of_layers;
  dzdy                         = (float*)p; p += sizeof(float) * number_of_layers;
  globaldy                     = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_x                   = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_y                   = (float*)p; p += sizeof(float) * number_of_layers;
  endpoint_z                   = (float*)p; p += sizeof(float) * number_of_layers;

  size = p - geometry;
}
