template<typename T, typename R>
__device__ int binary_search_first_candidate(
  const T* array,
  const uint array_size,
  const T& value,
  const float margin,
  const R& comparator)
{
  bool found = false;
  int l = 0;
  int r = array_size - 1;
  while (l < r) {
    const int m = (l + r) / 2;
    const auto array_element = array[m];

    // found |= std::abs(value - array_element) < margin;
    found |= comparator(value, array_element, m, margin);

    if (value - margin > array_element) {
      l = m + 1;
    }
    else {
      r = m;
    }
  }
  // found |= std::abs(value - array[l]) < margin;
  found |= comparator(value, array[l], l, margin);
  return found ? l : -1;
}