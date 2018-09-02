#include "Evaluator.cuh"

__global__ void catboost_evaluator(
  int* dev_tree_splits[],
  double* dev_leaf_values[],
  const int* dev_tree_sizes,
  float* dev_catboost_output,
  const unsigned char* dev_bin_features,
  const int dev_tree_num,
  const int dev_object_num,
  const int dev_bin_feature_num
) {
  const int objectId = blockIdx.x;
  const int blockSize = blockDim.x;
  if (objectId >= dev_object_num)
    return;
  int treeId = threadIdx.x;
  float sum = 0;

  while(treeId < dev_tree_num) {
    size_t index{};
    for (size_t depth = 0; depth < dev_tree_sizes[treeId]; ++depth) {
      index |= (dev_bin_features[objectId*dev_bin_feature_num+dev_tree_splits[treeId][depth]] << depth);
    }
    sum += dev_leaf_values[treeId][index];
    treeId += blockSize;
  }
  extern __shared__ float values[];
 
  int tid = threadIdx.x; 
  values[tid] = sum;

  __syncthreads();
  for(unsigned int s = blockSize; s > 0; s>>=1) {
    if (tid < s) {
      values[tid] += values[tid + s];
    }
    __syncthreads();
  }
  dev_catboost_output[objectId] = values[0];
 }