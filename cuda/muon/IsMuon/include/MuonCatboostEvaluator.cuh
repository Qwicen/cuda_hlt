#pragma once

#include "Handler.cuh"

__global__ void muon_catboost_evaluator(
  const float* dev_catboost_features,
  const float* dev_leaf_values,
  const int* dev_leaf_offsets,
  const float* dev_split_borders,
  const int* dev_split_features,
  const int* dev_tree_sizes,
  const int* dev_tree_offsets,
  const int n_trees,
  const int n_features,
  const int n_objects,
  float* dev_output
);

__device__ void warp_reduce(
  volatile float* sdata, 
  int tid
);

ALGORITHM(muon_catboost_evaluator, muon_catboost_evaluator_t)
