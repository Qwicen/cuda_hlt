#pragma once

#include "Constants.cuh"
#include <iostream>
#include <vector>

std::vector<float> run_kernel(std::vector<std::vector<float>>& features, Constants& constants);

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size();
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}