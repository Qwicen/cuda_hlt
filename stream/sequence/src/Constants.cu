#include "Constants.cuh"

void Constants::reserve_constants() {
  cudaCheck(cudaMalloc((void**)&dev_velo_module_zs, VeloTracking::n_modules * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_velo_candidate_ks, 9 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**)&dev_velo_sp_patterns, 256 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**)&dev_velo_sp_fx, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_velo_sp_fy, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_ut_dxDy, VeloUTTracking::n_layers * sizeof(float)));
}

void Constants::initialize_constants() {
  // Velo module constants
  const std::array<float, VeloTracking::n_modules> velo_module_zs = {-287.5, -275, -262.5, -250, -237.5, -225, -212.5, \
    -200, -137.5, -125, -62.5, -50, -37.5, -25, -12.5, 0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100, \
    112.5, 125, 137.5, 150, 162.5, 175, 187.5, 200, 212.5, 225, 237.5, 250, 262.5, 275, 312.5, 325, \
    387.5, 400, 487.5, 500, 587.5, 600, 637.5, 650, 687.5, 700, 737.5, 750};
  cudaCheck(cudaMemcpy(dev_velo_module_zs, velo_module_zs.data(), velo_module_zs.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Velo clustering candidate ks
  host_candidate_ks = {0, 0, 1, 2, 2, 3, 3, 3, 3};
  cudaCheck(cudaMemcpy(dev_velo_candidate_ks, host_candidate_ks.data(), host_candidate_ks.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

  // Velo clustering patterns
  // Fetch patterns and populate in GPU
  std::vector<uint8_t> sp_patterns (256, 0);
  std::vector<uint8_t> sp_sizes (256, 0);
  std::vector<float> sp_fx (512, 0);
  std::vector<float> sp_fy (512, 0);
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  cudaCheck(cudaMemcpy(dev_velo_sp_patterns, sp_patterns.data(), sp_patterns.size(), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fx, sp_fx.data(), sp_fx.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fy, sp_fy.data(), sp_fy.size() * sizeof(float), cudaMemcpyHostToDevice));

  // UT geometry constants
  // layer configuration: XUVX, U and V layers tilted by +/- 5 degrees = 0.087 radians
  host_ut_dxDy[0] = 0.;
  host_ut_dxDy[1] = 0.08748867;
  host_ut_dxDy[2] = -0.0874886;
  host_ut_dxDy[3] = 0.;

  cudaCheck(cudaMemcpy(dev_ut_dxDy, host_ut_dxDy, VeloUTTracking::n_layers * sizeof(float), cudaMemcpyHostToDevice));
}
