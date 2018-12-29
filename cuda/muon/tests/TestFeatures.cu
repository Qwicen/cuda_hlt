#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "TestFeatures.cuh"

SCENARIO( "Check closest hit works in case there is no extrapolation" ) {

    GIVEN( "Grid of hits with the same z coordinate at all stations" ) {

        // Hits initialization
        const int grid_size = 3;
        // Fill with 0, 1, 2, ...
        std::vector<int> muon_hit_tile(grid_size * grid_size);
        std::iota (std::begin(muon_hit_tile), std::end(muon_hit_tile), 0);
        // Grid initialization
        std::vector<float> muon_hit_x;
        std::vector<float> muon_hit_y;
        generateGrid(grid_size, muon_hit_x, muon_hit_y);
        // Fill with zeros
        std::vector<float> muon_hit_z(grid_size * grid_size);
        // Fill with 0, 1, 2, ...
        std::vector<float> muon_hit_dx(grid_size * grid_size);
        std::iota (std::begin(muon_hit_tile), std::end(muon_hit_tile), 0);
        // Fill with 0, 2, 4, ...
        std::vector<float> muon_hit_dy(grid_size * grid_size);
        int j = 0;
        for(std::vector<int>::iterator it = muon_hit_dy.begin() ; it != muon_hit_dy.end(); ++it){
            *it = j;
            j += 2;
        }
        // Fill with zeros (unused variable)
        std::vector<float> muon_hit_dz(grid_size * grid_size);

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

        // One event
        muon_hits_events.push_back(muon_hits);

        WHEN( "Track inside grid of hits and parallel to the axis OZ (x=0.9, y=0.9, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(0.9, 0.9, 0, 0, 0);
            MiniState *dev_track;
            cudaMalloc(&dev_track, 1 * sizeof(MiniState));
            cudaMemcpyAsync(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

            Initialise();

            muon_catboost_features_extraction<<<dim3(1, 4), 1>>>(
                dev_atomics_scifi,
                dev_scifi_track_hit_number,
                dev_qop,
                dev_track,
                dev_scifi_track_ut_indices,
                dev_muon_hits,
                dev_features
            );

            cudaMemcpy(host_features, dev_features, n_features * sizeof(float), cudaMemcpyDeviceToHost);

            THEN( "The closest hit at all stations is ( 1, 1). Indices: 8, 17, 26, 35" ) {
                const std::vector<int> closest_hits = {8, 17, 26, 35};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK(host_features[offset::RES_X + i_station] == -sqrt((8 + 9 * i_station) / 3);
                    CHECK(host_features[offset::RES_Y + i_station] == -sqrt(2 * (8 + 9 * i_station) / 3);
                }
            }
            cudaFree(dev_track);
            cudaFree(dev_muon_hits);
            cudaFree(dev_features);
            cudaFree(dev_qop);
            cudaFree(dev_atomics_scifi);
            cudaFree(dev_scifi_track_hit_number);
            cudaFree(dev_scifi_track_ut_indices);
        }

        WHEN( "Track equidistant from 4 hits and parallel to the axis OZ (x=0.5, y=0.5, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(0.5, 0.5, 0, 0, 0);
            MiniState *dev_track;
            cudaMalloc(&dev_track, 1 * sizeof(MiniState));
            cudaMemcpyAsync(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

            Initialise();

            muon_catboost_features_extraction<<<dim3(1, 4), 1>>>(
                dev_atomics_scifi,
                dev_scifi_track_hit_number,
                dev_qop,
                dev_track,
                dev_scifi_track_ut_indices,
                dev_muon_hits,
                dev_features
            );

            cudaMemcpy(host_features, dev_features, n_features * sizeof(float), cudaMemcpyDeviceToHost);

            THEN( "The closest hit at all stations is ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1). 
                Indices: (4, 13, 22, 31) or (7, 16, 25, 34) or (5, 14, 23, 32) or (8, 17, 26, 35)" ) {
                const std::vector<int> closest_hits = {8, 17, 26, 35};
                /*for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK(host_features[offset::RES_X + i_station] == -sqrt((8 + 9 * i_station) / 3);
                    CHECK(host_features[offset::RES_Y + i_station] == -sqrt(2 * (8 + 9 * i_station) / 3);
                }*/
            }
            cudaFree(dev_track);
            cudaFree(dev_muon_hits);
            cudaFree(dev_features);
            cudaFree(dev_qop);
            cudaFree(dev_atomics_scifi);
            cudaFree(dev_scifi_track_hit_number);
            cudaFree(dev_scifi_track_ut_indices);
        }
    }
    free(host_features);
}
