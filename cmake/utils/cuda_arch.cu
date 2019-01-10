#include <cuda_runtime.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[]){
  float min_cc = 3.0f;

  signed char c = '\0';
  bool latest_arch = false;
  while ((c = getopt(argc, argv, "lh?")) != -1) {
    switch (c) {
    case 'l':
      latest_arch = true;
      break;
    case '?':
    case 'h':
    default:
      std::cout << "usage: " << argv[0] << std::endl
                << " -l {select latest cuda architecture} " << std::endl;
      return -1;
    }
  }

  int n_devices = 0;
  int rc = cudaGetDeviceCount(&n_devices);
  if(rc != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return rc;
  }

  std::vector<std::pair<int, int>> arch(n_devices);

  for (int cd = 0; cd < n_devices; ++cd) {
    cudaDeviceProp dev;
    int rc = cudaGetDeviceProperties(&dev, cd);
    if(rc != cudaSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
      return rc;
    } else {
      arch[cd] = {dev.major, dev.minor};
    }
  }

  std::pair<int, int> best_cc{0, 0};
  if(latest_arch) {
    best_cc = *std::max_element(begin(arch), end(arch));
  } else {
    best_cc = *std::min_element(begin(arch), end(arch));
  }

  if((best_cc.first + best_cc.second / 10.f) < min_cc) {
    std::cout << "Min Compute Capability of "
              << std::fixed << std::setprecision(1) << min_cc << " required; "
              << best_cc.first << "." << best_cc.second << " found.";
    return 1;
  } else {
    std::cout << "sm_" << best_cc.first << best_cc.second;
    return 0;
  }
}
