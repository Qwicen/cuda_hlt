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
  uint32_t number_of_quarters_per_layer;
  uint32_t number_of_layers;
  uint32_t number_of_quarters;
  uint32_t* number_of_modules; //for each quarter
  uint32_t number_of_mats_per_module;
  uint32_t number_of_mats;
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
