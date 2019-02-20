#include <iterator>
#include <vector>
#include <numeric>
#include "MuonDefinitions.cuh"
#include "MuonFeaturesExtraction.cuh"

float* dev_features;
float* dev_qop;
int* dev_atomics_scifi;
uint* dev_scifi_track_hit_number;
uint* dev_scifi_track_ut_indices;
MiniState* dev_track;
const int n_features = 20;
const float MSFACTOR = 5.552176750308537;
const float COMMON_FACTOR = MSFACTOR * 0.23850119787527452 * 1; // 1 = qop
const float INVSQRT3 = 0.5773502691896258;
const float eps = 0.0001;

void dev_allocate_memory()
{
  cudaMalloc(&dev_track, 1 * sizeof(MiniState));
  cudaMalloc(&dev_features, 1 * n_features * sizeof(float));

  std::vector<float> host_qop = {1};
  cudaMalloc(&dev_qop, host_qop.size() * sizeof(float));
  cudaMemcpy(dev_qop, host_qop.data(), host_qop.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<int> host_atomics_scifi = {1, 0, 1};
  cudaMalloc((void**) &dev_atomics_scifi, host_atomics_scifi.size() * sizeof(int));
  cudaMemcpy(
    dev_atomics_scifi, host_atomics_scifi.data(), host_atomics_scifi.size() * sizeof(int), cudaMemcpyHostToDevice);

  std::vector<uint> host_scifi_track_hit_number = {0};
  cudaMalloc(&dev_scifi_track_hit_number, host_scifi_track_hit_number.size() * sizeof(uint));
  cudaMemcpy(
    dev_scifi_track_hit_number,
    host_scifi_track_hit_number.data(),
    host_scifi_track_hit_number.size() * sizeof(uint),
    cudaMemcpyHostToDevice);

  std::vector<uint> host_scifi_track_ut_indices = {42};
  cudaMalloc(&dev_scifi_track_ut_indices, host_scifi_track_ut_indices.size() * sizeof(uint));
  cudaMemcpy(
    dev_scifi_track_ut_indices,
    host_scifi_track_ut_indices.data(),
    host_scifi_track_ut_indices.size() * sizeof(uint),
    cudaMemcpyHostToDevice);
}

void dev_free_memory()
{
  cudaFree(dev_track);
  cudaFree(dev_features);
  cudaFree(dev_qop);
  cudaFree(dev_atomics_scifi);
  cudaFree(dev_scifi_track_hit_number);
  cudaFree(dev_scifi_track_ut_indices);
}

void generate_grid(const int n, std::vector<float>& x, std::vector<float>& y)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i - n / 2) != 0 || (j - n / 2) != 0) {
        y.push_back((i - n / 2) * 10);
        x.push_back((j - n / 2) * 10);
      }
    }
  }
}

Muon::HitsSoA construct_mock_muon_hit()
{
  // Hits initialization
  const int grid_size = 5;
  // Fill with 0, 1, 2, ...
  std::vector<int> muon_hit_tile(Muon::Constants::n_stations * (grid_size * grid_size - 1));
  std::iota(std::begin(muon_hit_tile), std::end(muon_hit_tile), 0);
  // Grid initialization
  std::vector<float> x, y;
  std::vector<float> muon_hit_x, muon_hit_y, muon_hit_z;
  generate_grid(grid_size, x, y);
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    std::vector<float> z(grid_size * grid_size - 1, (i_station + 1) * 100);
    muon_hit_x.insert(muon_hit_x.end(), x.begin(), x.end());
    muon_hit_y.insert(muon_hit_y.end(), y.begin(), y.end());
    muon_hit_z.insert(muon_hit_z.end(), z.begin(), z.end());
  }
  // Fill with 0, 1, 2, ...
  std::vector<float> muon_hit_dx(Muon::Constants::n_stations * (grid_size * grid_size - 1));
  std::iota(std::begin(muon_hit_dx), std::end(muon_hit_dx), 0);
  // Fill with 0, 0.5, 1, ...
  std::vector<float> muon_hit_dy(Muon::Constants::n_stations * (grid_size * grid_size - 1));
  float j = 0;
  for (std::vector<float>::iterator it = muon_hit_dy.begin(); it != muon_hit_dy.end(); ++it) {
    *it = j;
    j += 0.5;
  }
  // Fill with zeros (unused variable)
  std::vector<float> muon_hit_dz(Muon::Constants::n_stations * (grid_size * grid_size - 1));

  Muon::HitsSoA muon_hits;
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    muon_hits.number_of_hits_per_station[i_station] = grid_size * grid_size - 1;
    if (i_station == 0) {
      muon_hits.station_offsets[i_station] = 0;
    }
    else {
      muon_hits.station_offsets[i_station] =
        muon_hits.station_offsets[i_station - 1] + muon_hits.number_of_hits_per_station[i_station - 1];
    }
    int station_offset = muon_hits.station_offsets[i_station];
    for (int i_hit = 0; i_hit < muon_hits.number_of_hits_per_station[i_station]; i_hit++) {
      muon_hits.tile[station_offset + i_hit] = muon_hit_tile[station_offset + i_hit];

      muon_hits.x[station_offset + i_hit] = muon_hit_x[station_offset + i_hit];
      muon_hits.dx[station_offset + i_hit] = muon_hit_dx[station_offset + i_hit];
      muon_hits.y[station_offset + i_hit] = muon_hit_y[station_offset + i_hit];
      muon_hits.dy[station_offset + i_hit] = muon_hit_dy[station_offset + i_hit];
      muon_hits.z[station_offset + i_hit] = muon_hit_z[station_offset + i_hit];
      muon_hits.dz[station_offset + i_hit] = muon_hit_dz[station_offset + i_hit];
      // if uncrossed = 1 then time = delta_time
      muon_hits.uncrossed[station_offset + i_hit] = i_hit % 2;
      muon_hits.time[station_offset + i_hit] = 10;
      muon_hits.delta_time[station_offset + i_hit] = 10 + (1 - i_hit % 2) * 5;
      muon_hits.cluster_size[station_offset + i_hit] = 0;
    }
  }
  return muon_hits;
}

bool any_of(const std::vector<int> closest_hits, const int calculated_value, const int* values)
{
  std::vector<int> true_values;
  for (std::vector<int>::const_iterator it = closest_hits.begin(); it != closest_hits.end(); ++it) {
    true_values.push_back(values[*it]);
  }
  return std::any_of(
    true_values.cbegin(), true_values.cend(), [calculated_value](int i) { return i == calculated_value; });
}

std::vector<float> calculate_residual(
  const std::vector<int> closest_hits,
  const float extrapolation_x,
  const float* x,
  const float* dx,
  const float multiple_scattering_error)
{
  std::vector<float> residual;
  for (std::vector<int>::const_iterator it = closest_hits.begin(); it != closest_hits.end(); ++it) {
    const float value =
      (extrapolation_x - x[*it]) /
      sqrt(dx[*it] * dx[*it] * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error);
    residual.push_back(value);
  }
  return residual;
}
