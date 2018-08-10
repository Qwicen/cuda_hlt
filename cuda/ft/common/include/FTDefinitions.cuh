#pragma once

#include <stdint.h>
#include <vector>
#include <iostream>

/**
 * @brief FT geometry description typecast.
 */
struct FTGeometry {
  size_t size;
  uint32_t number_of_stations;
  uint32_t number_of_layers_per_station;
  uint32_t number_of_layers;
  uint32_t* number_of_modules; //for each layer
  float* dxdy;
  float* dzdy;
  float* globaldy;
  float* endpoint_x;
  float* endpoint_y;
  float* endpoint_z;

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
