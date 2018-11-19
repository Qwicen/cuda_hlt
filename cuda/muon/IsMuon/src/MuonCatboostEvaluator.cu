#include "MuonCatboostEvaluator.cuh"
/**
* Computes probability of being a muon.
* CatBoost uses oblivious trees as base predictors. In such trees same splitting criterion is used
across an entire level of a tree.
* In oblivious trees each leaf index can be encoded as a binary vector with length equal to a depth of the tree.
* First it represents each float feature as binary vector.
* Each binary element answers the question whether float feature more or less than corresponding threshold
* Then it uses binary features to calculate model predictions.
*
* CatBoost: gradient boosting with categorical features support:
* http://learningsys.org/nips17/assets/papers/paper_11.pdf
*/
__global__ void muon_catboost_evaluator(
  const float* const* dev_muon_catboost_borders,
  const float* dev_muon_catboost_features,
  const int* dev_muon_catboost_border_nums,
  const int* const* dev_muon_catboost_tree_splits,
  const int* dev_muon_catboost_feature_map,
  const int* dev_muon_catboost_border_map,
  const double* const* dev_muon_catboost_leaf_values,
  const int* dev_muon_catboost_tree_sizes,
  float* dev_muon_catboost_output,
  const int dev_muon_catboost_tree_num,
  const int dev_muon_catboost_object_num,
  const int dev_muon_catboost_bin_feature_num,
  const int dev_muon_catboost_float_feature_num
) {
  const int object_id = blockIdx.x;
  const int block_size = blockDim.x;
  if (object_id >= dev_muon_catboost_object_num)
    return;
  int tree_id = threadIdx.x;
  float sum = 0;
  const int object_offset = object_id * dev_muon_catboost_float_feature_num;
  while(tree_id < dev_muon_catboost_tree_num) {
    int index = 0;
    for (int depth = 0; depth < dev_muon_catboost_tree_sizes[tree_id]; ++depth) {
      const int split_num = dev_muon_catboost_tree_splits[tree_id][depth];
      const int feature_id = dev_muon_catboost_feature_map[split_num];
      const int border_id = dev_muon_catboost_border_map[split_num];
      const float feature_value = dev_muon_catboost_features[object_offset + feature_id];
      const float border = dev_muon_catboost_borders[feature_id][border_id];
      const int bin_feature = (int)(feature_value > border);
      index |= (bin_feature << depth);
    }
    sum += dev_muon_catboost_leaf_values[tree_id][index];
    tree_id += block_size;
  }
  extern __shared__ float values[];
 
  int tid = threadIdx.x; 
  values[tid] = sum;
   __syncthreads();
  for (unsigned int s=block_size/2; s>=32; s>>=1) {
    if (tid < s)
      values[tid] += values[tid + s];
    __syncthreads();
  }
  if (tid < 32) warp_reduce(values, tid);
  
  if (threadIdx.x == 0)
    dev_muon_catboost_output[object_id] = values[0];
}

__device__ void warp_reduce(
  volatile float* sdata, 
  int tid
) {
  for (unsigned int s=16; s>0; s>>=1)
    if( tid < s )
      sdata[tid] += sdata[tid + s];
}

