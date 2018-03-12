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