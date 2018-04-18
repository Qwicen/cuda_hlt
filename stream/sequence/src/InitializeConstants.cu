#include "../include/InitializeConstants.cuh"

void initializeConstants() {
  // Velo module constants
  const std::array<float, 52> velo_module_zs = {-287.5, -275, -262.5, -250, -237.5, -225, -212.5, \
    -200, -137.5, -125, -62.5, -50, -37.5, -25, -12.5, 0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100, \
    112.5, 125, 137.5, 150, 162.5, 175, 187.5, 200, 212.5, 225, 237.5, 250, 262.5, 275, 312.5, 325, \
    387.5, 400, 487.5, 500, 587.5, 600, 637.5, 650, 687.5, 700, 737.5, 750};
  cudaCheck(cudaMemcpyToSymbol(VeloTracking::velo_module_zs, velo_module_zs.data(), velo_module_zs.size() * sizeof(float)));

  // Clustering patterns
  // Fetch patterns and populate in GPU
  std::vector<uint8_t> sp_patterns (256, 0);
  std::vector<uint8_t> sp_sizes (256, 0);
  std::vector<float> sp_fx (512, 0);
  std::vector<float> sp_fy (512, 0);
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  cudaCheck(cudaMemcpyToSymbol(VeloClustering::sp_patterns, sp_patterns.data(), sp_patterns.size()));
  cudaCheck(cudaMemcpyToSymbol(VeloClustering::sp_sizes, sp_sizes.data(), sp_sizes.size()));
  cudaCheck(cudaMemcpyToSymbol(VeloClustering::sp_fx, sp_fx.data(), sp_fx.size() * sizeof(float)));
  cudaCheck(cudaMemcpyToSymbol(VeloClustering::sp_fy, sp_fy.data(), sp_fy.size() * sizeof(float)));
}
