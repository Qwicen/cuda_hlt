/*
*   Tests for calculation muon catbost features
*   How to run it
*   ./cuda/muon/TestFeatures
*/
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "TestFeatures.cuh"
using namespace Catch::Matchers;

SCENARIO( "Check closest hit works in case there is no extrapolation" ) {

    DevAllocateMemory();

    GIVEN( 
        "Grid of hits\n" 
        "There is 9 hits on each station with coordinates x,y: \n"
        "\t (-1, 1) - ( 0, 1) - ( 1, 1) \n"
        "\t (-1, 0) - ( 0, 0) - ( 1, 0) \n"
        "\t (-1,-1) - ( 0,-1) - ( 1,-1) \n"
        "z = i_station + 1 \n"
        "Hits indices on first station: \n"
        "\t 6	-    7	  -    8 \n"
        "\t 3	-    4	  -    5 \n"
        "\t 0	-    1	  -    2 \n"
        "and so on \n"
        "dx = index, dy = 2 * index, dz = 0 \n"
    ) {

        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = ConstructMockMuonHit();

        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        const float MSFACTOR = 5.552176750308537;
        const float COMMON_FACTOR = MSFACTOR * 0.23850119787527452 * 1; // 1 = qop
        const float INVSQRT3 = 0.5773502691896258;
        const float eps = 0.0001;
        float *host_features = (float*)malloc(1 * n_features * sizeof(float));

        WHEN( "Track inside grid of hits and parallel to the axis OZ (x=0.9, y=0.9, z=0, dx=0, dy=0)" ) {

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
                "Extrapolation of track:   \n"
                "\t station 0 - (0.9, 0.9) \n"
                "\t station 1 - (0.9, 0.9) \n"
                "\t station 2 - (0.9, 0.9) \n"
                "\t station 3 - (0.9, 0.9) \n"
                "Closest hits: \n"
                "\t station 0 - ( 1, 1), index = 8  \n"
                "\t station 1 - ( 1, 1), index = 17 \n"
                "\t station 2 - ( 1, 1), index = 26 \n"
                "\t station 3 - ( 1, 1), index = 35 \n"
                "Traveled distance: \n"
                "\t station 0 - 0   \n"
                "\t station 1 - 1   \n"
                "\t station 2 - 2   \n"
                "\t station 3 - 3   \n"
            ) {
                const std::vector<int> closest_hits = {8, 17, 26, 35};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                const std::vector<float> extrapolation_x = {0.9, 0.9, 0.9, 0.9};
                const std::vector<float> extrapolation_y = {0.9, 0.9, 0.9, 0.9};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }
        
        WHEN( "Track equidistant from 4 hits and parallel to the axis OZ (x=0.5, y=0.5, z=0, dx=0, dy=0)" ) {

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
                "Extrapolation of track:   \n"
                "\t station 0 - (0.5, 0.5) \n"
                "\t station 1 - (0.5, 0.5) \n"
                "\t station 2 - (0.5, 0.5) \n"
                "\t station 3 - (0.5, 0.5) \n"
                "Closest hits: \n"
                "\t station 0 - ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1), index =  4 or  7 or  5 or 8  \n"
                "\t station 1 - ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1), index = 13 or 16 or 14 or 17 \n"
                "\t station 2 - ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1), index = 22 or 25 or 23 or 26 \n"
                "\t station 3 - ( 0, 0) or ( 0, 1) or ( 1, 0) or ( 1, 1), index = 31 or 34 or 32 or 35 \n"
                "Traveled distance: \n"
                "\t station 0 - 0   \n"
                "\t station 1 - 1   \n"
                "\t station 2 - 2   \n"
                "\t station 3 - 3   \n"
            ) {
                const std::vector<int> closest_hits1 = {4, 13, 22, 31};
                const std::vector<int> closest_hits2 = {7, 16, 25, 34};
                const std::vector<int> closest_hits3 = {5, 14, 23, 32};
                const std::vector<int> closest_hits4 = {8, 17, 26, 35};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                const std::vector<float> extrapolation_x = {0.5, 0.5, 0.5, 0.5};
                const std::vector<float> extrapolation_y = {0.5, 0.5, 0.5, 0.5};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx1 = closest_hits1[i_station];
                    const int closest_idx2 = closest_hits2[i_station];
                    const int closest_idx3 = closest_hits3[i_station];
                    const int closest_idx4 = closest_hits4[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
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
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx1]) / 
                            sqrt(closest_idx1 * closest_idx1 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx2]) / 
                            sqrt(closest_idx2 * closest_idx2 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx3]) / 
                            sqrt(closest_idx3 * closest_idx3 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx4]) / 
                            sqrt(closest_idx4 * closest_idx4 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx1]) / 
                            sqrt(4 * closest_idx1 * closest_idx1 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx2]) / 
                            sqrt(4 * closest_idx2 * closest_idx2 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx3]) / 
                            sqrt(4 * closest_idx3 * closest_idx3 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps) ||
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx4]) / 
                            sqrt(4 * closest_idx4 * closest_idx4 * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }

        WHEN( "Track mathes hit and parallel to the axis OZ (x=1, y=-1, z=0, dx=0, dy=0)" ) {

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
                "Extrapolation of track:\n"
                "\t station 0 - ( 1,-1) \n"
                "\t station 1 - ( 1,-1) \n"
                "\t station 2 - ( 1,-1) \n"
                "\t station 3 - ( 1,-1) \n"
                "Closest hits: \n"
                "\t station 0 - ( 1,-1), index = 2  \n"
                "\t station 1 - ( 1,-1), index = 11 \n"
                "\t station 2 - ( 1,-1), index = 20 \n"
                "\t station 3 - ( 1,-1), index = 29 \n"
                "Traveled distance: \n"
                "\t station 0 - 0   \n"
                "\t station 1 - 1   \n"
                "\t station 2 - 2   \n"
                "\t station 3 - 3   \n"
            ) {
                const std::vector<int> closest_hits = {2, 11, 20, 29};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                const std::vector<float> extrapolation_x = {1, 1, 1, 1};
                const std::vector<float> extrapolation_y = {-1, -1, -1, -1};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }

        WHEN( "Track is far away from all hits and parallel to the axis OZ (x=1000, y=0.4, z=0, dx=0, dy=0)" ) {

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
                "Extrapolation of track:\n"
                "\t station 0 - (1000, 0.4) \n"
                "\t station 1 - (1000, 0.4) \n"
                "\t station 2 - (1000, 0.4) \n"
                "\t station 3 - (1000, 0.4) \n"
                "Closest hits: \n"
                "\t station 0 - ( 1, 0), index = 5  \n"
                "\t station 1 - ( 1, 0), index = 14 \n"
                "\t station 2 - ( 1, 0), index = 23 \n"
                "\t station 3 - ( 1, 0), index = 32 \n"
                "Traveled distance: \n"
                "\t station 0 - 0   \n"
                "\t station 1 - 1   \n"
                "\t station 2 - 2   \n"
                "\t station 3 - 3   \n"
            ) {
                const std::vector<int> closest_hits = {5, 14, 23, 32};
                const std::vector<float> trav_dist = {0, 1, 2, 3};
                const std::vector<float> extrapolation_x = {1000, 1000, 1000, 1000};
                const std::vector<float> extrapolation_y = {0.4, 0.4, 0.4, 0.4};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station],
                        WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }
        free(host_features);
        cudaFree(dev_muon_hits);
    }
    DevFreeMemory();
}

SCENARIO( "Check closest hit works in general case" ) {

    DevAllocateMemory();

    GIVEN( 
        "Grid of hits\n" 
        "There is 9 hits on each station with coordinates x,y: \n"
        "\t (-1, 1) - ( 0, 1) - ( 1, 1) \n"
        "\t (-1, 0) - ( 0, 0) - ( 1, 0) \n"
        "\t (-1,-1) - ( 0,-1) - ( 1,-1) \n"
        "z = i_station + 1 \n"
        "Hits indices on first station: \n"
        "\t 6	-    7	  -    8 \n"
        "\t 3	-    4	  -    5 \n"
        "\t 0	-    1	  -    2 \n"
        "and so on \n"
        "dx = index, dy = 2 * index, dz = 0 \n"
    ) {

        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = ConstructMockMuonHit();

        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        const float MSFACTOR = 5.552176750308537;
        const float COMMON_FACTOR = MSFACTOR * 0.23850119787527452 * 1; // 1 = qop
        const float INVSQRT3 = 0.5773502691896258;
        const float eps = 0.0001;
        float *host_features = (float*)malloc(1 * n_features * sizeof(float));

        WHEN( "Track inside grid of hits (x=-2.7, y=-2.7, z=0, dx=1, dy=1)" ) {

            // Track initialization
            MiniState track = MiniState(-2.7, -2.7, 0, 1, 1);
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
                "Extrapolation of track: \n"
                "\t station 1 - (-1.7,-1.7) \n"
                "\t station 2 - (-0.7,-0.7) \n"
                "\t station 3 - ( 0.3, 0.3) \n"
                "\t station 4 - ( 1.3, 1.3) \n"
                "Closest hits: \n"
                "\t station 1 - (-1,-1), index = 0 \n"
                "\t station 2 - (-1,-1), index = 9 \n"
                "\t station 3 - ( 0, 0), index = 22 \n"
                "\t station 4 - ( 1, 1), index = 35 \n"
                "Traveled distance: \n"
                "\t station 1 - 0 \n"
                "\t station 2 - sqrt(3) \n"
                "\t station 3 - sqrt(12) \n"
                "\t station 4 - sqrt(27) \n"
            ) {
                const std::vector<int> closest_hits = {0, 9, 22, 35};
                const std::vector<float> extrapolation_x = {-1.7, -0.7, 0.3, 1.3};
                const std::vector<float> extrapolation_y = {-1.7, -0.7, 0.3, 1.3};
                const std::vector<float> trav_dist = {0, sqrt(3.0f), sqrt(12.0f), sqrt(27.0f)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }

        WHEN( "Track inside grid of hits (x=-2.2, y=2.1, z=0, dx=1, dy=-0.5)" ) {

            // Track initialization
            MiniState track = MiniState(-2.2, 2.1, 0, 1, -0.5);
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
                "Extrapolation of track:    \n"
                "\t station 1 - (-1.2, 1.6) \n"
                "\t station 2 - (-0.2, 1.1) \n"
                "\t station 3 - ( 0.8, 0.6) \n"
                "\t station 4 - ( 1.8, 0.1) \n"
                "Closest hits: \n"
                "\t station 1 - (-1,-1), index = 6  \n"
                "\t station 2 - ( 0, 1), index = 16 \n"
                "\t station 3 - ( 1, 1), index = 26 \n"
                "\t station 4 - ( 1, 0), index = 32 \n"
                "Traveled distance: \n"
                "\t station 1 - 0   \n"
                "\t station 2 - 1.5 \n"
                "\t station 3 - 3   \n"
                "\t station 4 - sqrt(20.25) \n"
            ) {
                const std::vector<int> closest_hits = {6, 16, 26, 32};
                const std::vector<float> extrapolation_x = {-1.2, -0.2, 0.8, 1.8};
                const std::vector<float> extrapolation_y = { 1.6,  1.1, 0.6, 0.1};
                const std::vector<float> trav_dist = {0, 1.5, 3, sqrt(20.25f)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }

        WHEN( "Track is far away from hits (x=999, y=-2.7, z=0, dx=1, dy=1)" ) {

            // Track initialization
            MiniState track = MiniState(999, -2.7, 0, 1, 1);
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
                "Extrapolation of track:    \n"
                "\t station 1 - (1000,-1.7) \n"
                "\t station 2 - (1001,-0.7) \n"
                "\t station 3 - (1002, 0.3) \n"
                "\t station 4 - (1003, 1.3) \n"
                "Closest hits: \n"
                "\t station 1 - ( 1,-1), index = 2  \n"
                "\t station 2 - ( 1,-1), index = 11 \n"
                "\t station 3 - ( 1, 0), index = 23 \n"
                "\t station 4 - ( 1, 1), index = 35 \n"
                "Traveled distance:      \n"
                "\t station 1 - 0        \n"
                "\t station 2 - sqrt(3)  \n"
                "\t station 3 - sqrt(12) \n"
                "\t station 4 - sqrt(27) \n"
            ) {
                const std::vector<int> closest_hits = {2, 11, 23, 35};
                const std::vector<float> extrapolation_x = {1000, 1001, 1002, 1003};
                const std::vector<float> extrapolation_y = {-1.7, -0.7, 0.3, 1.3};
                const std::vector<float> trav_dist = {0, sqrt(3.0f), sqrt(12.0f), sqrt(27.0f)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float errMS = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(4 * closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + errMS * errMS), eps)
                    );
                }
            }
        }
        free(host_features);
        cudaFree(dev_muon_hits);
    }
    DevFreeMemory();
}