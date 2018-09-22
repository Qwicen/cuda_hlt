#pragma once

#include "math_constants.h" // PI

template<typename T>
__device__ int binary_search_first_candidate(
  const T* array,
  const uint array_size,
  const T& value,
  const float margin
) {
  bool found = false;
  int l = 0;
  int r = array_size - 1;
  while (l < r) {
    const int m = (l + r) / 2;
    const auto array_element = array[m];
    found |= std::abs(value - array_element) < margin;
    if (value - margin > array_element) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  found |= std::abs(value - array[l]) < margin;
  return found ? l : -1;
}

template<typename T>
__device__ int binary_search_second_candidate(
  const T* array,
  const uint array_size,
  const T& value,
  const float margin
) {
  int l = 0;
  int r = array_size - 1;
  while (l < r) {
    const int m = (l + r) / 2;
    if (value + margin > array[m]) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  const bool last_compatible = std::abs(value - array[l]) < margin;
  return last_compatible ? l+1 : l;
}

/**
 * @brief Calculate a single hit phi in odd sensor
 */
__device__ float hit_phi_odd(
  const float x,
  const float y
);

/**
 * @brief Calculate a single hit phi in even sensor
 */
__device__ float hit_phi_even(
  const float x,
  const float y
);
