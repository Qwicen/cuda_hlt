#include "Constants.cuh"

void Constants::reserve_constants()
{
  cudaCheck(cudaMalloc((void**) &dev_velo_module_zs, Velo::Constants::n_modules * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_velo_candidate_ks, 9 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_patterns, 256 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_fx, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_fy, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_ut_dxDy, UT::Constants::n_layers * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_tmva1, sizeof(SciFi::Tracking::TMVA)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_tmva2, sizeof(SciFi::Tracking::TMVA)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_constArrays, sizeof(SciFi::Tracking::Arrays)));
  cudaCheck(cudaMalloc(
    (void**) &dev_ut_region_offsets, (UT::Constants::n_layers * UT::Constants::n_regions_in_layer + 1) * sizeof(uint)));
  cudaCheck(cudaMalloc((void**) &dev_inv_clus_res, host_inv_clus_res.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations)));
  cudaCheck(cudaMalloc((void**) &dev_muon_foi, sizeof(Muon::Constants::FieldOfInterest)));
  cudaCheck(cudaMalloc((void**) &dev_muon_momentum_cuts, 3 * sizeof(float)));
}

void Constants::initialize_constants(
  const std::vector<float>& muon_field_of_interest_params
) {

  // Velo module constants
  const std::array<float, Velo::Constants::n_modules> velo_module_zs = {
    -287.5, -275,  -262.5, -250,  -237.5, -225,  -212.5, -200,  -137.5, -125,  -62.5, -50,   -37.5,
    -25,    -12.5, 0,      12.5,  25,     37.5,  50,     62.5,  75,     87.5,  100,   112.5, 125,
    137.5,  150,   162.5,  175,   187.5,  200,   212.5,  225,   237.5,  250,   262.5, 275,   312.5,
    325,    387.5, 400,    487.5, 500,    587.5, 600,    637.5, 650,    687.5, 700,   737.5, 750};
  cudaCheck(cudaMemcpy(
    dev_velo_module_zs, velo_module_zs.data(), velo_module_zs.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Velo clustering candidate ks
  host_candidate_ks = {0, 0, 1, 4, 4, 5, 5, 5, 5};
  cudaCheck(cudaMemcpy(
    dev_velo_candidate_ks,
    host_candidate_ks.data(),
    host_candidate_ks.size() * sizeof(uint8_t),
    cudaMemcpyHostToDevice));

  // Velo clustering patterns
  // Fetch patterns and populate in GPU
  std::vector<uint8_t> sp_patterns(256, 0);
  std::vector<uint8_t> sp_sizes(256, 0);
  std::vector<float> sp_fx(512, 0);
  std::vector<float> sp_fy(512, 0);
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  cudaCheck(cudaMemcpy(dev_velo_sp_patterns, sp_patterns.data(), sp_patterns.size(), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fx, sp_fx.data(), sp_fx.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fy, sp_fy.data(), sp_fy.size() * sizeof(float), cudaMemcpyHostToDevice));

  // UT geometry constants
  // layer configuration: XUVX, U and V layers tilted by +/- 5 degrees = 0.087 radians
  host_ut_dxDy = {0., 0.08748867, -0.0874886, 0.};
  cudaCheck(cudaMemcpy(dev_ut_dxDy, host_ut_dxDy.data(), host_ut_dxDy.size() * sizeof(float), cudaMemcpyHostToDevice));

  host_ut_region_offsets = {0, 84, 164, 248, 332, 412, 496, 594, 674, 772, 870, 950, 1048};
  cudaCheck(cudaMemcpy(
    dev_ut_region_offsets,
    host_ut_region_offsets.data(),
    host_ut_region_offsets.size() * sizeof(uint),
    cudaMemcpyHostToDevice));

  // SciFi constants
  SciFi::Tracking::TMVA host_tmva1;
  SciFi::Tracking::TMVA host_tmva2;
  SciFi::Tracking::TMVA1_Init(host_tmva1);
  SciFi::Tracking::TMVA2_Init(host_tmva2);
  SciFi::Tracking::Arrays host_constArrays;

  cudaCheck(cudaMemcpy(dev_scifi_tmva1, &host_tmva1, sizeof(SciFi::Tracking::TMVA), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_scifi_tmva2, &host_tmva2, sizeof(SciFi::Tracking::TMVA), cudaMemcpyHostToDevice));
  cudaCheck(
    cudaMemcpy(dev_scifi_constArrays, &host_constArrays, sizeof(SciFi::Tracking::Arrays), cudaMemcpyHostToDevice));

  host_inv_clus_res = {1 / 0.05, 1 / 0.08, 1 / 0.11, 1 / 0.14, 1 / 0.17, 1 / 0.20, 1 / 0.23, 1 / 0.26, 1 / 0.29};
  cudaCheck(
    cudaMemcpy(dev_inv_clus_res, &host_inv_clus_res, host_inv_clus_res.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Kalman filter constants.
  ParKalmanFilter::KalmanParametrizations host_kalman_params;
  host_kalman_params.SetParameters("../cuda/kalman/params/FT6x2", ParKalmanFilter::Polarity::Down);
  cudaCheck(cudaMemcpy(
    dev_kalman_params, &host_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations), cudaMemcpyHostToDevice));

  // Muon constants
  Muon::Constants::FieldOfInterest host_muon_foi;
  const float* foi_iterator = muon_field_of_interest_params.data();
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_a_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_a_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_b_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_b_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_c_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_c_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;
  }
  cudaCheck(cudaMemcpy(dev_muon_momentum_cuts, &Muon::Constants::momentum_cuts, 3 * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_muon_foi, &host_muon_foi, sizeof(Muon::Constants::FieldOfInterest), cudaMemcpyHostToDevice));
}

void Constants::initialize_ut_decoding_constants(const std::vector<char>& ut_geometry)
{
  const UTGeometry geometry(ut_geometry);

  // Offset for each station / layer
  const std::array<uint, UT::Constants::n_layers + 1> offsets {host_ut_region_offsets[0],
                                                               host_ut_region_offsets[3],
                                                               host_ut_region_offsets[6],
                                                               host_ut_region_offsets[9],
                                                               host_ut_region_offsets[12]};
  auto current_sector_offset = 0;
  host_unique_x_sector_offsets[current_sector_offset];
  host_unique_x_sector_layer_offsets[0] = 0;

  for (int i = 0; i < UT::Constants::n_layers; ++i) {
    const auto offset = offsets[i];
    const auto size = offsets[i + 1] - offsets[i];

    // Copy elements into xs vector
    std::vector<float> xs;
    std::copy_n(geometry.p0X + offset, size, std::back_inserter(xs));

    // Create permutation
    std::vector<int> permutation(xs.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    // Sort permutation according to xs
    std::stable_sort(
      permutation.begin(), permutation.end(), [&xs](const int& a, const int& b) { return xs[a] < xs[b]; });

    // Iterate the permutation, incrementing the counter when the element changes.
    // Calculate unique elements
    std::vector<int> permutation_repeated;
    auto current_element = xs[permutation[0]];
    int current_index = 0;
    int number_of_unique_elements = 1;

    for (auto p : permutation) {
      // Allow for a configurable window of error
      constexpr float accepted_error_window = 2.f;
      if (std::abs(current_element - xs[p]) > accepted_error_window) {
        current_element = xs[p];
        current_index++;
        number_of_unique_elements++;
      }
      permutation_repeated.emplace_back(current_index);
    }

    // Calculate final permutation into unique elements
    std::vector<int> unique_permutation;
    for (int j = 0; j < size; ++j) {
      auto it = std::find(permutation.begin(), permutation.end(), j);
      auto position = it - permutation.begin();
      unique_permutation.emplace_back(permutation_repeated[position]);
    }

    // Some printouts in case we want to debug
    // if (logger::ll.verbosityLevel >= logger::debug) {
    //   for (int j=0; j<size; ++j) {
    //     debug_cout << j << ", " << geometry.p0X[offset + j] << ", " << permutation[j] << ", "
    //       << permutation_repeated[j] << ", " << unique_permutation[j] << std::endl;
    //   }
    //   std::vector<float> unique_elements (number_of_unique_elements);
    //   for (int j=0; j<size; ++j) {
    //     const int index = unique_permutation[j];
    //     unique_elements[index] = xs[j];
    //   }
    //   debug_cout << "Unique elements: " << number_of_unique_elements << std::endl;
    //   for (int j=0; j<number_of_unique_elements; ++j) {
    //     debug_cout << unique_elements[j] << ", ";
    //   }
    //   debug_cout << std::endl;
    // }

    // Fill in host_unique_sector_xs
    std::vector<float> temp_unique_elements(number_of_unique_elements);
    for (int j = 0; j < size; ++j) {
      const int index = unique_permutation[j];
      temp_unique_elements[index] = xs[j];
    }
    for (int j = 0; j < number_of_unique_elements; ++j) {
      host_unique_sector_xs.emplace_back(temp_unique_elements[j]);
    }

    // Fill in host_unique_x_sector_offsets
    for (auto p : unique_permutation) {
      host_unique_x_sector_offsets.emplace_back(current_sector_offset + p);
    }

    // Fill in host_unique_x_sectors
    current_sector_offset += number_of_unique_elements;
    host_unique_x_sector_layer_offsets[i + 1] = current_sector_offset;
  }

  // Some debug printouts
  // if (logger::ll.verbosityLevel >= logger::debug) {
  //   debug_cout << "Unique X sectors:";
  //   for (auto i : host_unique_x_sector_layer_offsets) {
  //     debug_cout << i << std::endl;
  //   }
  //   debug_cout << std::endl << "Unique X sector permutation:" << std::endl;
  //   for (auto i : host_unique_x_sector_offsets) {
  //     debug_cout << i << ", ";
  //   }
  //   debug_cout << std::endl;
  // }

  // Populate device constant into global memory
  cudaCheck(
    cudaMalloc((void**) &dev_unique_x_sector_layer_offsets, host_unique_x_sector_layer_offsets.size() * sizeof(uint)));
  cudaCheck(cudaMalloc((void**) &dev_unique_x_sector_offsets, host_unique_x_sector_offsets.size() * sizeof(uint)));
  cudaCheck(cudaMalloc((void**) &dev_unique_sector_xs, host_unique_sector_xs.size() * sizeof(float)));
  cudaCheck(cudaMemcpy(
    dev_unique_x_sector_layer_offsets,
    host_unique_x_sector_layer_offsets.data(),
    host_unique_x_sector_layer_offsets.size() * sizeof(uint),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_unique_x_sector_offsets,
    host_unique_x_sector_offsets.data(),
    host_unique_x_sector_offsets.size() * sizeof(uint),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_unique_sector_xs,
    host_unique_sector_xs.data(),
    host_unique_sector_xs.size() * sizeof(float),
    cudaMemcpyHostToDevice));
}

void Constants::initialize_geometry_constants(
  const std::vector<char>& velopix_geometry,
  const std::vector<char>& ut_boards,
  const std::vector<char>& ut_geometry,
  const std::vector<char>& ut_magnet_tool,
  const std::vector<char>& scifi_geometry)
{
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**) &dev_velo_geometry, velopix_geometry.size()));
  cudaCheck(cudaMemcpy(dev_velo_geometry, velopix_geometry.data(), velopix_geometry.size(), cudaMemcpyHostToDevice));

  // Populate UT boards and geometry
  cudaCheck(cudaMalloc((void**) &dev_ut_boards, ut_boards.size()));
  cudaCheck(cudaMemcpy(dev_ut_boards, ut_boards.data(), ut_boards.size(), cudaMemcpyHostToDevice));

  cudaCheck(cudaMalloc((void**) &dev_ut_geometry, ut_geometry.size()));
  cudaCheck(cudaMemcpy(dev_ut_geometry, ut_geometry.data(), ut_geometry.size(), cudaMemcpyHostToDevice));

  // Populate UT magnet tool values
  cudaCheck(cudaMalloc((void**) &dev_ut_magnet_tool, ut_magnet_tool.size()));
  cudaCheck(cudaMemcpy(dev_ut_magnet_tool, ut_magnet_tool.data(), ut_magnet_tool.size(), cudaMemcpyHostToDevice));

  // Populate FT geometry
  cudaCheck(cudaMalloc((void**) &dev_scifi_geometry, scifi_geometry.size()));
  cudaCheck(cudaMemcpy(dev_scifi_geometry, scifi_geometry.data(), scifi_geometry.size(), cudaMemcpyHostToDevice));
  host_scifi_geometry = scifi_geometry.data();
}

void Constants::initialize_muon_catboost_model_constants(
  const int n_trees,
  const std::vector<int>& tree_depths,
  const std::vector<int>& tree_offsets,
  const std::vector<float>& leaf_values,
  const std::vector<int>& leaf_offsets,
  const std::vector<float>& split_borders,
  const std::vector<int>& split_features
) {
  muon_catboost_n_trees = n_trees;
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_split_features, split_features.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_split_borders, split_borders.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_leaf_values, leaf_values.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_tree_depths, tree_depths.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_tree_offsets, tree_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_leaf_offsets, leaf_offsets.size() * sizeof(int)));

  cudaCheck(cudaMemcpy(
    dev_muon_catboost_split_features,
    split_features.data(),
    split_features.size() * sizeof(int),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_split_borders,
    split_borders.data(),
    split_borders.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_leaf_values, leaf_values.data(), leaf_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_tree_depths, tree_depths.data(), tree_depths.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_tree_offsets, tree_offsets.data(), tree_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_leaf_offsets, leaf_offsets.data(), leaf_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
}
