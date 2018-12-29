#include<iterator>
#include "MuonDefinitions.cuh"
#include "MuonFeaturesExtraction.cuh"

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
            muon_hits.tile[station_offset + i_hit] = tile[i_hit];

            muon_hits.x[station_offset + i_hit] = x[i_hit];
            muon_hits.dx[station_offset + i_hit] = dx[i_hit];
            muon_hits.y[station_offset + i_hit] = y[i_hit];
            muon_hits.dy[station_offset + i_hit] = dy[i_hit];
            muon_hits.z[station_offset + i_hit] = z[i_hit];
            muon_hits.dz[station_offset + i_hit] = dz[i_hit];

            muon_hits.uncrossed[station_offset + i_hit] = 0;
            muon_hits.time[station_offset + i_hit]          = 1000;
            muon_hits.delta_time[station_offset + i_hit]    = 1000;
            muon_hits.cluster_size[station_offset + i_hit]  = 1000;
        }
    }
    return muon_hits;
}

std::vector<Muon::HitsSoA> muon_hits_events;
Muon::HitsSoA *dev_muon_hits;

const int n_features = 20;
float *host_features = (float*)malloc(1 * n_features * sizeof(float));// 210
float *dev_features;
float *dev_qop;
int *dev_atomics_scifi;
uint *dev_scifi_track_hit_number;
uint *dev_scifi_track_ut_indices;

void Initialise() {
    
    cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
    cudaMemcpyAsync(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

    // Features initialization
    cudaMalloc(&dev_features, 1 * n_features * sizeof(float));

    // QoP initialization
    std::vector<float> host_qop;
    host_qop.push_back(1);
    cudaMalloc(&dev_qop, host_qop.size() * sizeof(float));
    cudaMemcpyAsync(dev_qop, host_qop.data(), host_qop.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Atomics scifi
    std::vector<int> host_atomics_scifi;
    host_atomics_scifi.push_back(-42);
    cudaMalloc((void**)&dev_atomics_scifi, host_atomics_scifi.size() * sizeof(int));
    cudaMemcpyAsync(dev_atomics_scifi, host_atomics_scifi.data(), host_atomics_scifi.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Scifi Hit Number
    std::vector<uint> host_scifi_track_hit_number;
    host_scifi_track_hit_number.push_back(-42);
    cudaMalloc(&dev_scifi_track_hit_number, host_scifi_track_hit_number.size() * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_hit_number, host_scifi_track_hit_number.data(), host_scifi_track_hit_number.size() * sizeof(uint), cudaMemcpyHostToDevice);

    // Scifi track ut indices
    std::vector<uint> host_scifi_track_ut_indices;
    host_scifi_track_ut_indices.push_back(-42);
    cudaMalloc(&dev_scifi_track_ut_indices, host_scifi_track_ut_indices.size() * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_ut_indices, host_scifi_track_ut_indices.data(), host_scifi_track_ut_indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
}

void generateGrid(const int n, std::vector<float> &x, std::vector<float> &y) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			y.push_back(i - n/2);
			x.push_back(j - n/2);
		}
	}
}