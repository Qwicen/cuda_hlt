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
  const float* dev_muon_catboost_features,
  float* dev_muon_catboost_output,
  const float* dev_muon_catboost_leaf_values,
  const int* dev_muon_catboost_leaf_offsets,
  const float* dev_muon_catboost_split_borders,
  const int* dev_muon_catboost_split_features,
  const int* dev_muon_catboost_tree_sizes,
  const int* dev_muon_catboost_tree_offsets,
  const int n_trees
) {
  const int object_id = blockIdx.x;
  const int block_size = blockDim.x;
  int tree_id = threadIdx.x;
  float sum = 0;
  
  const int object_offset = object_id * Muon::Constants::n_catboost_features;

  while(tree_id < n_trees) {
    int index = 0;
    const int tree_offset = dev_muon_catboost_tree_offsets[tree_id];
    for (int depth = 0; depth < dev_muon_catboost_tree_sizes[tree_id]; ++depth) {
      const int feature_id = dev_muon_catboost_split_features[tree_offset + depth];
      const float feature_value = dev_muon_catboost_features[object_offset + feature_id];
      const float border = dev_muon_catboost_split_borders[tree_offset + depth];
      const int bin_feature = (int)(feature_value > border);
      index |= (bin_feature << depth);
    }
    sum += dev_muon_catboost_leaf_values[dev_muon_catboost_leaf_offsets[tree_id] + index];
    tree_id += block_size;
  }
  __shared__ float values[256];
 
  int tid = threadIdx.x;
  values[tid] = sum;
   __syncthreads();
  for (unsigned int s=block_size/2; s > 0; s>>=1) {
    if (tid < s)
      values[tid] += values[tid + s];
    __syncthreads();
  }
  
  if (threadIdx.x == 0)
     dev_muon_catboost_output[object_id] = values[0];
    
}
