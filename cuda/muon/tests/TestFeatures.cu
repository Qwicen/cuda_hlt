#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "TestFeatures.cuh"

SCENARIO( "Muon catboost features evaluation" ) {

    GIVEN( "Track, hits, qop" ) {

        // Track initialization
        MiniState track = MiniState(0, 0, 1, 1, 1);
        MiniState *dev_track;
        cudaMalloc(&dev_track, 1 * sizeof(MiniState));
        cudaMemcpyAsync(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

        // Hits initialization
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

        Initialise();

        WHEN( "features calculated" ) {
            
            BENCHMARK( "muon_catboost_features_extraction kernel call" ) {
                muon_catboost_features_extraction<<<dim3(10, 4), 1>>>(
                    dev_atomics_scifi,
                    dev_scifi_track_hit_number,
                    dev_qop,
                    dev_track,
                    dev_scifi_track_ut_indices,
                    dev_muon_hits,
                    dev_features
                );
            }
            
            cudaMemcpy(host_features, dev_features, n_features * sizeof(float), cudaMemcpyDeviceToHost);

            THEN( "the result is reasonable" ) {

                for (int i_event = 0; i_event < n_events; i_event ++) {
                    for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                        const int closest_idx = 0;
                        CHECK(host_features[offset::DTS + i_station] == muon_hits_events[i_event].delta_time[closest_idx]);
                        CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[i_event].time[closest_idx]);
                        CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[i_event].uncrossed[closest_idx] == 2);
                        
                        //CHECK(host_features[offset::CROSS + i_station] == muon_hits_events[i_event].RES_X[0]);
                        //CHECK(host_features[offset::CROSS + i_station] == muon_hits_events[i_event].RES_Y[0]);
                    }
                }
            }
        }

        cudaFree(dev_track);
        cudaFree(dev_muon_hits);
        cudaFree(dev_features);
        cudaFree(dev_qop);
        cudaFree(dev_atomics_scifi);
        cudaFree(dev_scifi_track_hit_number);
        cudaFree(dev_scifi_track_ut_indices);
        free(host_features);
        free(host_qop);
        free(host_atomics_scifi);
        free(host_scifi_track_hit_number);
        free(host_scifi_track_ut_indices);
    }
}
