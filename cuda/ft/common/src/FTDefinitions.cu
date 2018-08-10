#include "FTDefinitions.cuh"

/*                     ---Geometry format---
| name                         | type     | size               |
|------------------------------|----------|--------------------|
| number_of_stations           | uint32_t | 1                  |
| number_of_layers_per_station | uint32_t | 1                  |
| number_of_quarters_per_layer | uint32_t | 1                  |
| number_of_modules            | uint32_t | number_of_quarters |
| number_of_mats_per_module    | uint32_t | 1                  |
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
