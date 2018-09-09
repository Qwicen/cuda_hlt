__global__ void gen_bin_features(
  float* dev_borders[],
  float* dev_features[],
  const int* dev_border_nums,
  unsigned char* dev_bin_features,
  const int dev_object_num,
  const int dev_bin_feature_num
);