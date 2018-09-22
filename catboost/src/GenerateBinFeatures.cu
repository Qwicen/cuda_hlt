#include "GenerateBinFeatures.cuh"
#include <stdio.h>
__global__ void gen_bin_features(
  const float* const* dev_borders,
  const float* const* dev_features,
  const int* dev_border_nums,
  unsigned char* dev_bin_features,
  const int dev_object_num,
  const int dev_bin_feature_num
) {
  const int objectId = blockIdx.x;
  if (objectId >= dev_object_num)
    return;
  int FloatFeatureId = threadIdx.x;

  const float floatVal = dev_features[objectId][FloatFeatureId];
  int idx = objectId*dev_bin_feature_num;
  for(int i = 0; i < FloatFeatureId; i++) {
    idx += dev_border_nums[i];
  }
  for(int j = 0; j < dev_border_nums[FloatFeatureId]; j++) {
    float border = dev_borders[FloatFeatureId][j];
    dev_bin_features[idx+j] = (unsigned char)(floatVal > border);
  }
}