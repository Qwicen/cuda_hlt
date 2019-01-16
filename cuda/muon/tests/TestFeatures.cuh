#include <iterator>
#include <vector>
#include "MuonDefinitions.cuh"
#include "MuonFeaturesExtraction.cuh"

float *dev_features;
float *dev_qop;
int *dev_atomics_scifi;
uint *dev_scifi_track_hit_number;
uint *dev_scifi_track_ut_indices;
MiniState *dev_track;
const int n_features = 20;
const float MSFACTOR = 5.552176750308537;
const float COMMON_FACTOR = MSFACTOR * 0.23850119787527452 * 1; // 1 = qop
const float INVSQRT3 = 0.5773502691896258;
const float eps = 0.0001;

void DevAllocateMemory() {
    cudaMalloc(&dev_track, 1 * sizeof(MiniState));
    cudaMalloc(&dev_features, 1 * n_features * sizeof(float));

    std::vector<float> host_qop = {1};
    cudaMalloc(&dev_qop, host_qop.size() * sizeof(float));
    cudaMemcpy(dev_qop, host_qop.data(), host_qop.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<int> host_atomics_scifi = {1};
    cudaMalloc((void**)&dev_atomics_scifi, host_atomics_scifi.size() * sizeof(int));
    cudaMemcpy(dev_atomics_scifi, host_atomics_scifi.data(), host_atomics_scifi.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<uint> host_scifi_track_hit_number = {0};
    cudaMalloc(&dev_scifi_track_hit_number, host_scifi_track_hit_number.size() * sizeof(uint));
    cudaMemcpy(dev_scifi_track_hit_number, host_scifi_track_hit_number.data(), host_scifi_track_hit_number.size() * sizeof(uint), cudaMemcpyHostToDevice);

    std::vector<uint> host_scifi_track_ut_indices = {42};
    cudaMalloc(&dev_scifi_track_ut_indices, host_scifi_track_ut_indices.size() * sizeof(uint));
    cudaMemcpy(dev_scifi_track_ut_indices, host_scifi_track_ut_indices.data(), host_scifi_track_ut_indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
}

void DevFreeMemory() {
    cudaFree(dev_track);
    cudaFree(dev_features);
    cudaFree(dev_qop);
    cudaFree(dev_atomics_scifi);
    cudaFree(dev_scifi_track_hit_number);
    cudaFree(dev_scifi_track_ut_indices);
}

void generateGrid(const int n, std::vector<float> &x, std::vector<float> &y) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			y.push_back(i - n/2);
			x.push_back(j - n/2);
		}
	}
}

Muon::HitsSoA ConstructMockMuonHit() {
    // Hits initialization
    const int grid_size = 3;
    // Fill with 0, 1, 2, ...
    std::vector<int> muon_hit_tile(Muon::Constants::n_stations * grid_size * grid_size);
    std::iota (std::begin(muon_hit_tile), std::end(muon_hit_tile), 0);
    // Grid initialization
    std::vector<float> x, y;
    std::vector<float> muon_hit_x, muon_hit_y, muon_hit_z;
    generateGrid(grid_size, x, y);
    for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
        std::vector<float> z(grid_size * grid_size, i_station + 1);
        muon_hit_x.insert(muon_hit_x.end(), x.begin(), x.end());
        muon_hit_y.insert(muon_hit_y.end(), y.begin(), y.end());
        muon_hit_z.insert(muon_hit_z.end(), z.begin(), z.end());
    }
    // Fill with 0, 1, 2, ...
    std::vector<float> muon_hit_dx(Muon::Constants::n_stations * grid_size * grid_size);
    std::iota (std::begin(muon_hit_dx), std::end(muon_hit_dx), 0);
    // Fill with 0, 2, 4, ...
    std::vector<float> muon_hit_dy(Muon::Constants::n_stations * grid_size * grid_size);
    float j = 0;
    for(std::vector<float>::iterator it = muon_hit_dy.begin() ; it != muon_hit_dy.end(); ++it){
        *it = j;
        j += 2;
    }
    // Fill with zeros (unused variable)
    std::vector<float> muon_hit_dz(Muon::Constants::n_stations * grid_size * grid_size);

    Muon::HitsSoA muon_hits;
    for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
        muon_hits.number_of_hits_per_station[i_station] = grid_size * grid_size;
        if (i_station == 0) {
            muon_hits.station_offsets[i_station] = 0;
        } else {
            muon_hits.station_offsets[i_station] = muon_hits.station_offsets[i_station - 1] + 
                                                   muon_hits.number_of_hits_per_station[i_station - 1];
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

            muon_hits.uncrossed[station_offset + i_hit] = 0;
            muon_hits.time[station_offset + i_hit]          = 1000;
            muon_hits.delta_time[station_offset + i_hit]    = 1000;
            muon_hits.cluster_size[station_offset + i_hit]  = 1000;
        }
    }
    return muon_hits;
}

bool any_of(
    const std::vector<int> closest_hits, 
    const int calculated_value, 
    const int* values
) {
    std::vector<int> true_values;
    for(std::vector<int>::const_iterator it = closest_hits.begin() ; it != closest_hits.end(); ++it){
        true_values.push_back(values[*it]);
    }
    return std::any_of(true_values.cbegin(), true_values.cend(), [calculated_value](int i){ return i == calculated_value; });
}

std::vector<float> calculate_res(
    const std::vector<int> closest_hits, 
    const float extrapolation,
    const float* x,
    const float* dx,
    const float multiple_scattering_error
) {
    std::vector<float> res;
    for(std::vector<int>::const_iterator it = closest_hits.begin() ; it != closest_hits.end(); ++it){
        const float value = (extrapolation - x[*it]) / 
            sqrt(dx[*it] * dx[*it] * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error);
        res.push_back(value);
    }
    return res;
}