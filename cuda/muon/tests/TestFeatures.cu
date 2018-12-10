#define CATCH_CONFIG_MAIN
#include "catch.hpp"
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

SCENARIO( "Muon catboost features evaluation" ) {

    GIVEN( "Track, hits, qop" ) {

        // Track initialization
        MiniState track = MiniState(0, 0, 1, 1, 1);
        MiniState *dev_track;
        cudaMalloc(&dev_track, 1 * sizeof(MiniState));
        cudaMemcpyAsync(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);


        // Hits initialization
        const int n_events = 1;
        std::vector<Muon::HitsSoA> muon_hits_events;
        std::vector<int> muon_hit_tile = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_x = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_dx = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_y = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_dy = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_z = {1, 2, 3, 4, 5};
        std::vector<float> muon_hit_dz = {1, 2, 3, 4, 5};

        Muon::HitsSoA muon_hits = ConstructMockMuonHit(
            muon_hit_tile.size(), 
            muon_hit_tile.data(),
            muon_hit_x.data(), 
            muon_hit_dx.data(),
            muon_hit_y.data(), 
            muon_hit_dy.data(),
            muon_hit_z.data(), 
            muon_hit_dz.data()
        );
        for (int i = 0; i < n_events; i++) {
            muon_hits_events.push_back(muon_hits);
        }

        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpyAsync(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        
        // Features initialization
        const int n_features = 20;
        float *host_features = (float*)malloc(n_features * sizeof(float));
        float *dev_features;
        cudaMalloc(&dev_features, n_features * sizeof(float));


        // QoP initialization
        float *host_qop = (float*)malloc(1 * sizeof(float));
        float *dev_qop;
        host_qop[0] = 5;
        cudaMalloc(&dev_qop, 1);
        cudaMemcpyAsync(dev_qop, &host_qop, 1 * sizeof(float), cudaMemcpyHostToDevice);

        WHEN( "features calculated" ) {
            
            BENCHMARK( "muon_catboost_features_extraction kernel call" ) {
                muon_catboost_features_extraction<<<4,1>>>(
                    dev_track,
                    dev_muon_hits,
                    dev_qop,
                    dev_features
                );
            }
            
            cudaMemcpy(host_features, dev_features, n_features * sizeof(float), cudaMemcpyDeviceToHost);

            THEN( "the result is reasonable" ) {

                for (int i_event = 0; i_event < n_events; i_event ++) {
                    for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                        const int closest_idx = 0;
                        REQUIRE(host_features[offset::DTS + i_station] == muon_hits_events[i_event].delta_time[closest_idx]);
                        REQUIRE(host_features[offset::TIMES + i_station] == muon_hits_events[i_event].time[closest_idx]);
                        REQUIRE(host_features[offset::CROSS + i_station] + muon_hits_events[i_event].uncrossed[closest_idx] == 2);
                        
                        //REQUIRE(host_features[offset::CROSS + i_station] == muon_hits_events[i_event].RES_X[0]);
                        //REQUIRE(host_features[offset::CROSS + i_station] == muon_hits_events[i_event].RES_Y[0]);
                    }
                }
            }
        }

        cudaFree(dev_track);
        cudaFree(dev_muon_hits);
        cudaFree(dev_features);
        cudaFree(dev_qop);
        free(host_features);
        free(host_qop);
    }
}
