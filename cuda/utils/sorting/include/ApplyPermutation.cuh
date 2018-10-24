/*! 
 *  \brief     apply_permutation sorting tool.
 *  \author    Daniel Hugo Campora Perez
 *  \author    Dorothea vom Bruch
 *  \date      2018
 */

#pragma once

#include "CudaCommon.h"
#include <cassert>

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__host__ __device__ void apply_permutation(
  uint* permutation,
  const uint hit_start,
  const uint number_of_hits,
  T* prev_container,
  T* new_container)
{
  // Apply permutation across all hits
  FOR_STATEMENT(uint, i, number_of_hits) {
    const auto hit_index_global = permutation[hit_start + i];
    new_container[hit_start + i] = prev_container[hit_index_global];
  }
}
