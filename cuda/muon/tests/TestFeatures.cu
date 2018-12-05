#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "MuonDefinitions.cuh"
#include "MuonFeaturesExtraction.cuh"

Muon::HitsSoA ConstructMockMuonHit(int n_hits) {
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
            muon_hits.tile[station_offset + i_hit] = station_offset * 100 + i_hit;

            muon_hits.x[station_offset + i_hit] = 1;
            muon_hits.dx[station_offset + i_hit] = 2;
            muon_hits.y[station_offset + i_hit] = 3;
            muon_hits.dy[station_offset + i_hit] = 4;
            muon_hits.z[station_offset + i_hit] = 5;
            muon_hits.dz[station_offset + i_hit] = 6;

            muon_hits.uncrossed[station_offset + i_hit] = 0;
            muon_hits.time[station_offset + i_hit] = 5;
            muon_hits.delta_time[station_offset + i_hit] = 2;
            muon_hits.cluster_size[station_offset + i_hit] = 2;
        }
    }
    return muon_hits;
}

SCENARIO( "Muon catboost features evaluation" ) {

    GIVEN( "Track, hits, qop" ) {

        MiniState track = MiniState(0, 0, 1, 1, 1);
        MiniState *dev_track;
        cudaMalloc(&dev_track, 1 * sizeof(MiniState));
        cudaMemcpyAsync(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = ConstructMockMuonHit(5);
        muon_hits_events.push_back(muon_hits);
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, 1 * sizeof(Muon::HitsSoA));
        cudaMemcpyAsync(dev_muon_hits, &muon_hits_events[0], 1 * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        float *host_features = (float*)malloc(20 * sizeof(float));
        float *dev_features;
        cudaMalloc(&dev_features, 20 * sizeof(float));

        float *host_qop = (float*)malloc(1 * sizeof(float));
        float *dev_qop;
        host_qop[0] = 5;
        cudaMalloc(&dev_qop, 1);
        cudaMemcpyAsync(dev_qop, &host_qop, 1 * sizeof(float), cudaMemcpyHostToDevice);

        REQUIRE( muon_hits_events.size() == 1 );

        WHEN( "features calculated" ) {
            
            BENCHMARK( "muon_catboost_features_extraction kernel call" ) {
                muon_catboost_features_extraction<<<4,1>>>(
                    dev_track,
                    dev_muon_hits,
                    dev_qop,
                    dev_features
                );
            }
            
            cudaMemcpy(host_features, dev_features, 20 * sizeof(float), cudaMemcpyDeviceToHost);

            THEN( "the result is reasonable" ) {
                
                REQUIRE(host_features[0] == muon_hits_events[0].delta_time[0]);
                REQUIRE(host_features[1] == muon_hits_events[0].delta_time[1]);
                REQUIRE(host_features[2] == muon_hits_events[0].delta_time[2]);
                REQUIRE(host_features[3] == muon_hits_events[0].delta_time[3]);

                REQUIRE(host_features[4] == muon_hits_events[0].time[0]);
                REQUIRE(host_features[5] == muon_hits_events[0].time[1]);
                REQUIRE(host_features[6] == muon_hits_events[0].time[2]);
                REQUIRE(host_features[7] == muon_hits_events[0].time[3]);
            }
        }
    }
}
