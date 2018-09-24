#include "Evaluator.cuh"
#include <stdio.h>
__global__ void catboost_evaluator(
  const int* const* dev_tree_splits,
  const double* const* dev_leaf_values,
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
    int index{};
    for (int depth = 0; depth < dev_tree_sizes[treeId]; ++depth) {
      const int obj_shift = objectId * dev_bin_feature_num;
      const int split_num = dev_tree_splits[treeId][depth];
      index |= (dev_bin_features[obj_shift + split_num] << depth);
    }
    sum += dev_leaf_values[treeId][index];
    treeId += blockSize;
  }
  extern __shared__ float values[];
 
  int tid = threadIdx.x; 
  values[tid] = sum;

  __syncthreads();
  for (unsigned int s=blockSize/2; s>=32; s>>=1) {
    if (tid < s)
      values[tid] += values[tid + s];
    __syncthreads();
  }
  if (tid < 32) warpReduce(values, tid);
  
  if (threadIdx.x == 0)
    dev_catboost_output[objectId] = values[0];
}

__device__ void warpReduce(
  volatile float* sdata, 
  int tid
) {
  for (unsigned int s=16; s>0; s>>=1)
    if( tid < s )
      sdata[tid] += sdata[tid + s];
}