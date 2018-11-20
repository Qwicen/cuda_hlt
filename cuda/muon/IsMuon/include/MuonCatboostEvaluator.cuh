#pragma once

#include "Handler.cuh"

__global__ void muon_catboost_evaluator(
//  const float* const* dev_muon_catboost_borders,
//  const float* dev_muon_catboost_features,
//  const int* dev_muon_catboost_border_nums,
//  const int* const* dev_muon_catboost_tree_splits,
//  const int* dev_muon_catboost_feature_map,
//  const int* dev_muon_catboost_border_map,
//  const double* const* dev_muon_catboost_leaf_values,
//  const int* dev_muon_catboost_tree_sizes,
  float* dev_muon_catboost_output,
//  const int dev_muon_catboost_tree_num,
  const int dev_muon_catboost_object_num//,
//  const int dev_muon_catboost_bin_feature_num,
//  const int dev_muon_catboost_float_feature_num
);

__device__ void warp_reduce(
  volatile float* sdata, 
  int tid
);

ALGORITHM(muon_catboost_evaluator, muon_catboost_evaluator_t)
