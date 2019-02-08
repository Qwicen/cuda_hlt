#pragma once

template<typename T>
__device__ int binary_search_leftmost(const T* array, const uint array_size, const T& value)
{
  int l = 0;
  int r = array_size - 1;
  while (l < r) {
    const int m = (l + r) / 2;
    const auto array_element = array[m];
    if (value > array_element) {
      l = m + 1;
    }
    else {
      r = m;
    }
  }
  return l;
}
