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

const int n_events = 1;
std::vector<Muon::HitsSoA> muon_hits_events;
Muon::HitsSoA *dev_muon_hits;

const int n_features = 20;
float *host_features = (float*)malloc(n_features * sizeof(float));
float *dev_features;

float *host_qop = (float*)malloc(1 * sizeof(float));
float *dev_qop;

int *host_atomics_scifi = (int*)malloc(1 * sizeof(int));
int *dev_atomics_scifi;

uint *host_scifi_track_hit_number = (uint*)malloc(1 * sizeof(uint));
uint *dev_scifi_track_hit_number;

uint *host_scifi_track_ut_indices = (uint*)malloc(1 * sizeof(uint));
uint *dev_scifi_track_ut_indices;

void Initialise() {
    
    cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
    cudaMemcpyAsync(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

    // Features initialization
    cudaMalloc(&dev_features, n_features * sizeof(float));

    // QoP initialization
    host_qop[0] = 5;
    cudaMalloc(&dev_qop, 1 * sizeof(float));
    cudaMemcpyAsync(dev_qop, &host_qop, 1 * sizeof(float), cudaMemcpyHostToDevice);

    // Atomics scifi
    cudaMalloc(&dev_atomics_scifi, 1 * sizeof(int));
    cudaMemcpyAsync(dev_atomics_scifi, &host_atomics_scifi, 1 * sizeof(int), cudaMemcpyHostToDevice);

    // Scifi Hit Number
    cudaMalloc(&dev_scifi_track_hit_number, 1 * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_hit_number, &host_scifi_track_hit_number, 1 * sizeof(uint), cudaMemcpyHostToDevice);

    // Scifi track ut indices
    cudaMalloc(&dev_scifi_track_ut_indices, 1 * sizeof(uint));
    cudaMemcpyAsync(dev_scifi_track_ut_indices, &host_scifi_track_ut_indices, 1 * sizeof(uint), cudaMemcpyHostToDevice);
}