__global__ void gen_bin_features(
  const float* const* dev_borders,
  const float* const* dev_features,
  const int* dev_border_nums,
  unsigned char* dev_bin_features,
  const int dev_object_num,
  const int dev_bin_feature_num
);