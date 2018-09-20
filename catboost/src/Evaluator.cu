#include "Evaluator.cuh"
#include <stdio.h>
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
      const int obj_shift = objectId * dev_bin_feature_num;
      const int split_num = dev_tree_splits[treeId][depth];
      index |= (dev_bin_features[obj_shift + split_num] << depth);
    }
    sum += dev_leaf_values[treeId][index];
    treeId += blockSize;
  }
  __shared__ float values[32];
 
  int tid = threadIdx.x; 
  values[tid] = sum;

  __syncthreads();
  warpReduce(values, tid);

  dev_catboost_output[objectId] = values[0];
}

__device__ void warpReduce(
  volatile float* sdata, 
  int tid
) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}