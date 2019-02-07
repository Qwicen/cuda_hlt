#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"
#include "ArgumentsMuon.cuh"

__global__ void muon_catboost_evaluator(
  const float* dev_muon_catboost_features,
  float* dev_muon_catboost_output,
  const float* dev_muon_catboost_leaf_values,
  const int* dev_muon_catboost_leaf_offsets,
  const float* dev_muon_catbost_split_borders,
  const int* dev_muon_catboost_split_features,
  const int* dev_muon_catboost_tree_sizes,
  const int* dev_muon_catboost_tree_offsets,
  const int n_trees
);

ALGORITHM(muon_catboost_evaluator, muon_catboost_evaluator_t,
  ARGUMENTS(dev_muon_catboost_features,
    dev_muon_catboost_output))
