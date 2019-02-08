#pragma once

#include <vector>
#include <stdint.h>

void cache_sp_patterns(
  std::vector<unsigned char>& sp_patterns,
  std::vector<unsigned char>& sp_sizes,
  std::vector<float>& sp_fx,
  std::vector<float>& sp_fy);
