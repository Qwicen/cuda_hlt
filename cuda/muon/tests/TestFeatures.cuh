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

const int n_events = 10;
std::vector<Muon::HitsSoA> muon_hits_events;
Muon::HitsSoA *dev_muon_hits;

const int n_features = 20;
float *host_features = (float*)malloc(210 * n_features * sizeof(float));
float *dev_features;
float *dev_qop;
int *dev_atomics_scifi;
uint *dev_scifi_track_hit_number;
uint *dev_scifi_track_ut_indices;

void Initialise() {
    
    cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
    cudaMemcpyAsync(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

    // Features initialization
    cudaMalloc(&dev_features, 210 * n_features * sizeof(float));

    // QoP initialization
    std::ifstream file("dev_scifi_qop");
    std::vector<float> host_qop;
    if (file) {
        float value;
        while (file >> value) {
            host_qop.push_back(value);
        }
    }
    cudaMalloc(&dev_qop, host_qop.size() * sizeof(float));
    cudaMemcpyAsync(dev_qop, host_qop.data(), host_qop.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Atomics scifi
    file = std::ifstream("dev_atomics_scifi");
    std::vector<int> host_atomics_scifi;
    if (file) {
        int value;
        while (file >> value) {
            host_atomics_scifi.push_back(value);
        }
    }
    cudaMalloc((void**)&dev_atomics_scifi, host_atomics_scifi.size() * sizeof(int));
    cudaMemcpyAsync(dev_atomics_scifi, host_atomics_scifi.data(), host_atomics_scifi.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Scifi Hit Number
    file = std::ifstream("dev_scifi_track_hit_number");
    std::vector<uint> host_scifi_track_hit_number;
    if (file) {
        int value;
        while (file >> value) {
            host_scifi_track_hit_number.push_back(value);
        }
    }
    cudaMalloc(&dev_scifi_track_hit_number, host_scifi_track_hit_number.size() * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_hit_number, host_scifi_track_hit_number.data(), host_scifi_track_hit_number.size() * sizeof(uint), cudaMemcpyHostToDevice);

    // Scifi track ut indices
    file = std::ifstream("dev_ut_indices");
    std::vector<uint> host_scifi_track_ut_indices;
    if (file) {
        int value;
        while (file >> value) {
            host_scifi_track_ut_indices.push_back(value);
        }
    }
    cudaMalloc(&dev_scifi_track_ut_indices, host_scifi_track_ut_indices.size() * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_ut_indices, host_scifi_track_ut_indices.data(), host_scifi_track_ut_indices.size() * sizeof(uint), cudaMemcpyHostToDevice);
}