#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "TestFeatures.cuh"
using namespace Catch::Matchers;

SCENARIO( "Check closest hit works in case there is no extrapolation" ) {

    DevAllocateMemory();

    GIVEN( "Grid of hits" ) {

        std::vector<Muon::HitsSoA> muon_hits_events;
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
            std::vector<float> z(grid_size * grid_size, i_station);
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

        Muon::HitsSoA muon_hits = ConstructMockMuonHit(
            grid_size * grid_size, 
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
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        const float c = 0.23850119787527452 * 5.552176750308537;
        const float INVSQRT3 = 0.5773502691896258;
        float *host_features = (float*)malloc(1 * n_features * sizeof(float));

        WHEN( "Track inside grid of hits and parallel to the axis OZ (x=0.9, y=0.9, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(0.9, 0.9, 0, 0, 0);
            cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

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

            THEN(
                "Extrapolation at all stations is (0.9, 0.9).\n"
                "The closest hit at all stations is ( 1, 1).\n"
                "Indices: 8, 17, 26, 35.\n"
                "Traveled distance is 0, 1, 2, 3.\n"
            ) {
                const std::vector<int> closest_hits = {8, 17, 26, 35};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = c * 1 * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((0.9 - muon_hits_events[0].x[closest_idx]) / sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((0.9 - muon_hits_events[0].y[closest_idx]) / sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                }
            }
        }
        
        WHEN( "Track equidistant from 4 hits and parallel to the axis OZ (x=0.5, y=0.5, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(0.5, 0.5, 0, 0, 0);
            cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

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

            THEN( 
                "Extrapolation at all stations is (0.5, 0.5).\n"
                "The closest hit at all stations is ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1).\n" 
                "Indices: (4, 13, 22, 31) or (7, 16, 25, 34) or (5, 14, 23, 32) or (8, 17, 26, 35).\n"
                "Traveled distance is 0, 1, 2, 3.\n"
            ) {
                const std::vector<int> closest_hits1 = {4, 13, 22, 31};
                const std::vector<int> closest_hits2 = {7, 16, 25, 34};
                const std::vector<int> closest_hits3 = {5, 14, 23, 32};
                const std::vector<int> closest_hits4 = {8, 17, 26, 35};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx1 = closest_hits1[i_station];
                    const int closest_idx2 = closest_hits2[i_station];
                    const int closest_idx3 = closest_hits3[i_station];
                    const int closest_idx4 = closest_hits4[i_station];
                    const float errMS = c * 1 * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK((
                        host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx1] || 
                        host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx2] ||
                        host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx3] ||
                        host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx4]
                    ));
                    CHECK((
                        host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx1] ||
                        host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx2] ||
                        host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx3] ||
                        host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx4]
                    ));
                    CHECK((
                        host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx1] == 2 ||
                        host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx2] == 2 ||
                        host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx3] == 2 ||
                        host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx4] == 2
                    ));
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((0.5 - muon_hits_events[0].x[closest_idx1]) / sqrt(closest_idx1 * closest_idx1 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].x[closest_idx2]) / sqrt(closest_idx2 * closest_idx2 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].x[closest_idx3]) / sqrt(closest_idx3 * closest_idx3 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].x[closest_idx4]) / sqrt(closest_idx4 * closest_idx4 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((0.5 - muon_hits_events[0].y[closest_idx1]) / sqrt(4 * closest_idx1 * closest_idx1 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].y[closest_idx2]) / sqrt(4 * closest_idx2 * closest_idx2 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].y[closest_idx3]) / sqrt(4 * closest_idx3 * closest_idx3 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005) ||
                        WithinAbs((0.5 - muon_hits_events[0].y[closest_idx4]) / sqrt(4 * closest_idx4 * closest_idx4 * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005)
                    );
                }
            }
        }

        WHEN( "Track mathes hit and parallel to the axis OZ (x=1, y=-1, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(1, -1, 0, 0, 0);
            cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

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

            THEN( 
                "Extrapolation at all stations is ( 1,-1).\n"
                "The closest hit at all stations is ( 1,-1).\n"
                "Indices: 2, 11, 20, 29.\n"
                "Traveled distance is 0, 1, 2, 3.\n"
            ) {
                const std::vector<int> closest_hits = {2, 11, 20, 29};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = c * 1 * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((1 - muon_hits_events[0].x[closest_idx]) / sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((-1 - muon_hits_events[0].y[closest_idx]) / sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                }
            }
        }

        WHEN( "Track is far away from all hits and parallel to the axis OZ (x=1000, y=0.4, dx=0, dy=0, z=0)" ) {

            // Track initialization
            MiniState track = MiniState(1000, 0.4, 0, 0, 0);
            cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

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

            THEN(
                "Extrapolation at all stations is (1000, 0.4).\n"
                "The closest hit at all stations is ( 1, 0).\n" 
                "Indices: 5, 14, 23, 32.\n"
                "Traveled distance is 0, 1, 2, 3.\n"
            ) {
                const std::vector<int> closest_hits = {5, 14, 23, 32};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = c * 1 * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((1000 - muon_hits_events[0].x[closest_idx]) / sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                    CHECK_THAT(host_features[offset::RES_Y + i_station],
                        WithinAbs((0.4 - muon_hits_events[0].y[closest_idx]) / sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), 0.005));
                }
            }
        }
        free(host_features);
        cudaFree(dev_muon_hits);
    }
    DevFreeMemory();
}
