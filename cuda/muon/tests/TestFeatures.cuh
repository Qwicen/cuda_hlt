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

// Initialise each station with same hits
Muon::HitsSoA ConstructMockMuonHit(
    int n_hits,
    int* tile,
    float* x, 
    float* dx,
    float* y, 
    float* dy,
    float* z, 
    float* dz
) {
    Muon::HitsSoA muon_hits;
    for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
        muon_hits.number_of_hits_per_station[i_station] = n_hits;
        if (i_station == 0) {
            muon_hits.station_offsets[i_station] = 0;
        } else {
            muon_hits.station_offsets[i_station] = muon_hits.station_offsets[i_station - 1] + 
                                                   muon_hits.number_of_hits_per_station[i_station - 1];
        }
        int station_offset = muon_hits.station_offsets[i_station];
        for (int i_hit = 0; i_hit < muon_hits.number_of_hits_per_station[i_station]; i_hit++) {
            muon_hits.tile[station_offset + i_hit] = tile[station_offset + i_hit];

            muon_hits.x[station_offset + i_hit] = x[station_offset + i_hit];
            muon_hits.dx[station_offset + i_hit] = dx[station_offset + i_hit];
            muon_hits.y[station_offset + i_hit] = y[station_offset + i_hit];
            muon_hits.dy[station_offset + i_hit] = dy[station_offset + i_hit];
            muon_hits.z[station_offset + i_hit] = z[station_offset + i_hit];
            muon_hits.dz[station_offset + i_hit] = dz[station_offset + i_hit];

            muon_hits.uncrossed[station_offset + i_hit] = 0;
            muon_hits.time[station_offset + i_hit]          = 1000;
            muon_hits.delta_time[station_offset + i_hit]    = 1000;
            muon_hits.cluster_size[station_offset + i_hit]  = 1000;
        }
    }
    return muon_hits;
}

void generateGrid(const int n, std::vector<float> &x, std::vector<float> &y) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			y.push_back(i - n/2);
			x.push_back(j - n/2);
		}
	}
}