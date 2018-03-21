#pragma once

#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <iostream>
#include "cuda_runtime.h"

/// For sanity check of input
#define NUMBER_OF_SENSORS 52

/**
 * @brief Macro to check cuda calls.
 */
#define cudaCheck(stmt) {                                \
  cudaError_t err = stmt;                                \
  if (err != cudaSuccess){                               \
    std::cerr << "Failed to run " << #stmt << std::endl; \
    std::cerr << cudaGetErrorString(err) << std::endl;   \
    throw std::invalid_argument("cudaCheck failed");     \
  }                                                      \
}

/**
 * @brief Velo geometry description typecast.
 */
struct VeloGeometry {
  size_t size;
  uint32_t number_of_sensors;
  uint32_t number_of_sensor_columns;
  uint32_t number_of_sensors_per_module;
  uint32_t chip_columns;
  float pixel_size;
  double* local_x;
  double* x_pitch;
  float* ltg;

  /**
   * @brief Typecast from std::vector.
   */
  VeloGeometry(const std::vector<char>& geometry) {
    const char* p = geometry.data();

    number_of_sensors            = *((uint32_t*)p); p += sizeof(uint32_t);
    number_of_sensor_columns     = *((uint32_t*)p); p += sizeof(uint32_t);
    number_of_sensors_per_module = *((uint32_t*)p); p += sizeof(uint32_t);
    chip_columns     = *((uint32_t*)p); p += sizeof(uint32_t);
    local_x          = (double*) p; p += sizeof(double) * number_of_sensor_columns;
    x_pitch          = (double*) p; p += sizeof(double) * number_of_sensor_columns;
    pixel_size       = *((float*)p); p += sizeof(float);
    ltg              = (float*) p; p += sizeof(float) * 16 * number_of_sensors;
    
    size = p - geometry.data();

    if (size != geometry.size()) {
      std::cout << "Size mismatch for geometry" << std::endl;
    }
  }

  /**
   * @brief Just typecast, no size check.
   */
  VeloGeometry(
    const char* geometry
  ) {
    const char* p = geometry;

    number_of_sensors            = *((uint32_t*)p); p += sizeof(uint32_t);
    number_of_sensor_columns     = *((uint32_t*)p); p += sizeof(uint32_t);
    number_of_sensors_per_module = *((uint32_t*)p); p += sizeof(uint32_t);
    chip_columns     = *((uint32_t*)p); p += sizeof(uint32_t);
    local_x          = (double*) p; p += sizeof(double) * number_of_sensor_columns;
    x_pitch          = (double*) p; p += sizeof(double) * number_of_sensor_columns;
    pixel_size       = *((float*)p); p += sizeof(float);
    ltg              = (float*) p; p += sizeof(float) * 16 * number_of_sensors;
    
    size = p - geometry;
  }
};

// Maybe for the future:
// More efficient Velo format (not used atm)
// 
//   size_t size;
//   uint32_t number_of_raw_banks;
//   uint32_t* sensor_index;
//   uint32_t* sp_count;
//   uint32_t* sp_word;
//   

/**
 * @brief Velo raw event format typecast.
 */
struct VeloRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  char* payload;

  VeloRawEvent(
    const char* event,
    const unsigned int event_size
  ) {
    const char* p = event;

    number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
    raw_bank_offset = (uint32_t*) p; p += number_of_raw_banks * sizeof(uint32_t);
    payload = (char*) p;

    // Sanity check
    unsigned int last_nsp = 0, offset = 0;
    for (unsigned int i=0; i<number_of_raw_banks; ++i) {
      uint32_t* raw_bank_payload = (uint32_t*) (payload + raw_bank_offset[i]);
      uint32_t nsp = raw_bank_payload[1];

      if (i!=(number_of_raw_banks-1) && raw_bank_offset[i+1] != raw_bank_offset[i] + (2 + nsp) * sizeof(uint32_t)) {
        std::cout << "Warning: Unexpected VeloRawEvent offset" << std::endl;
      }

      last_nsp = nsp;
    }
    
    if (event_size != sizeof(uint32_t)
      + number_of_raw_banks * sizeof(uint32_t)
      + raw_bank_offset[number_of_raw_banks-1]
      + (2 + last_nsp) * sizeof(uint32_t)) {
      std::cout << "Warning: Size mismatch for VeloRawEvent" << std::endl;
    }
  }
};

struct VeloRawBank {
  uint32_t sensor_index;
  uint32_t sp_count;
  uint32_t* sp_word;

  VeloRawBank(const char* raw_bank) {
    const char* p = raw_bank;

    sensor_index = *((uint32_t*)p); p += sizeof(uint32_t);
    sp_count = *((uint32_t*)p); p += sizeof(uint32_t);
    sp_word = (uint32_t*) p;
  }
};

/**
 * @brief Struct to typecast events.
 */
struct EventInfo {
  size_t size;
  uint32_t numberOfModules;
  uint32_t numberOfHits;
  float* module_Zs;
  uint32_t* module_hitStarts;
  uint32_t* module_hitNums;
  uint32_t* hit_IDs;
  float* hit_Xs;
  float* hit_Ys;
  float* hit_Zs;

  EventInfo() = default;

  EventInfo(const std::vector<char>& event) {
    char* input = (char*) event.data();

    numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
    numberOfHits     = *((uint32_t*)input); input += sizeof(uint32_t);
    module_Zs        = (float*)input; input += sizeof(float) * numberOfModules;
    module_hitStarts = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    module_hitNums   = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - event.data();
  }

  EventInfo(const char* event) {
    char* input = (char*) event;
    
    numberOfModules  = *((uint32_t*)input); input += sizeof(uint32_t);
    numberOfHits     = *((uint32_t*)input); input += sizeof(uint32_t);
    module_Zs        = (float*)input; input += sizeof(float) * numberOfModules;
    module_hitStarts = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    module_hitNums   = (uint32_t*)input; input += sizeof(uint32_t) * numberOfModules;
    hit_IDs          = (uint32_t*)input; input += sizeof(uint32_t) * numberOfHits;
    hit_Xs           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Ys           = (float*)  input; input += sizeof(float)   * numberOfHits;
    hit_Zs           = (float*)  input; input += sizeof(float)   * numberOfHits;

    size = input - (char*) event;
  }
};
