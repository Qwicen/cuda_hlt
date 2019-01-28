#include "catch.hpp"
#include "MuonFeaturesExtraction.test.cuh"

SCENARIO( "Track is parallel to the axis OZ" ) {
    dev_allocate_memory();

    GIVEN( 
        "Grid of mock hits\n" 
        "There is 24 hits on each station with coordinates x,y: \n"
        "\t (-20, 20) - (-10, 20) - ( 0, 20) - ( 10, 20) - ( 20, 20)\n"
        "\t (-20, 10) - (-10, 10) - ( 0, 10) - ( 10, 10) - ( 20, 10)\n"
        "\t (-20,  0) - (-10,  0) -          - ( 10,  0) - ( 20,  0)\n"
        "\t (-20,-10) - (-10,-10) - ( 0,-10) - ( 10,-10) - ( 20,-10)\n"
        "\t (-20,-20) - (-10,-20) - ( 0,-20) - ( 10,-20) - ( 20,-20)\n"
        "z = 100, 200, 300, 400 for stations 1, 2, 3, 4 respectively\n"
        "Hits indices for the first station: \n"
        "\t 19	-   20	  -   21    -   22    -   23\n"
        "\t 14	-   15	  -   16    -   17    -   18\n"
        "\t 10	-   11	  -         -   12    -   13\n"
        "\t 5	-    6	  -    7    -    8    -    9\n"
        "\t 0	-    1	  -    2    -    3    -    4\n"
        "and so on \n"
        "Let dx = index, dy = index / 2, dz = 0\n"
    ) {
        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = construct_mock_muon_hit();
        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        float *host_features = (float*)malloc(1 * n_features * sizeof(float));

        WHEN( "Track parallel to the axis OZ (x=9, y=9, z=0, dx=0, dy=0)" ) {
            MiniState track = MiniState(9, 9, 0, 0, 0);
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
                "\t station 1 - (9, 9)  \n"
                "\t station 2 - (9, 9)  \n"
                "\t station 3 - (9, 9)  \n"
                "\t station 4 - (9, 9)  \n"
                "Closest hits: \n"
                "\t station 1 - ( 10, 10), index = 17  \n"
                "\t station 2 - ( 10, 10), index = 41 \n"
                "\t station 3 - ( 10, 10), index = 65 \n"
                "\t station 4 - ( 10, 10), index = 89 \n"
                "Traveled distance: \n"
                "\t station 1 -   0 \n"
                "\t station 2 - 100 \n"
                "\t station 3 - 200 \n"
                "\t station 4 - 300 \n"
            ) {
                const std::vector<int> closest_hits = {17, 41, 65, 89};
                const std::vector<float> trav_dist = {0, 100, 200, 300};
                const std::vector<float> extrapolation_x = {9, 9, 9, 9};
                const std::vector<float> extrapolation_y = {9, 9, 9, 9};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }
        
        WHEN( "Track parallel to the axis OZ (x=5, y=5, z=0, dx=0, dy=0)" ) {
            MiniState track = MiniState(5, 5, 0, 0, 0);
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
                "\t station 1 - (5, 5)  \n"
                "\t station 2 - (5, 5)  \n"
                "\t station 3 - (5, 5)  \n"
                "\t station 4 - (5, 5)  \n"
                "Closest hits: \n"
                "\t station 1 - ( 0, 10) or ( 10, 0) or ( 10, 10), index = 16 or 12 or 17  \n"
                "\t station 2 - ( 0, 10) or ( 10, 0) or ( 10, 10), index = 40 or 36 or 41 \n"
                "\t station 3 - ( 0, 10) or ( 10, 0) or ( 10, 10), index = 64 or 60 or 65 \n"
                "\t station 4 - ( 0, 10) or ( 10, 0) or ( 10, 10), index = 88 or 84 or 89 \n"
                "Traveled distance: \n"
                "\t station 1 -   0 \n"
                "\t station 2 - 100 \n"
                "\t station 3 - 200 \n"
                "\t station 4 - 300 \n"
            ) {
                const std::vector<std::vector<int>> closest_hits = {{16, 12, 17}, {40, 36, 41}, {64, 60, 65}, {88, 84, 89}};
                const std::vector<float> trav_dist = {0, 100, 200, 300};
                const std::vector<float> extrapolation_x = {5, 5, 5, 5};
                const std::vector<float> extrapolation_y = {5, 5, 5, 5};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(
                        any_of(closest_hits[i_station], host_features[offset::DTS + i_station], 
                            muon_hits_events[0].delta_time) == true
                    );
                    CHECK(
                        any_of(closest_hits[i_station], host_features[offset::TIMES + i_station], 
                            (int*) muon_hits_events[0].time) == true
                    );
                    CHECK(
                        any_of(closest_hits[i_station], 2 - host_features[offset::CROSS + i_station], 
                            muon_hits_events[0].uncrossed) == true
                    );
                    const std::vector<float> true_res_x = calculate_residual(
                        closest_hits[i_station],
                        extrapolation_x[i_station],
                        muon_hits_events[0].x,
                        muon_hits_events[0].dx,
                        multiple_scattering_error
                    );
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        Catch::Matchers::WithinAbs(true_res_x[0], eps) ||
                        Catch::Matchers::WithinAbs(true_res_x[1], eps) ||
                        Catch::Matchers::WithinAbs(true_res_x[2], eps)
                    );
                    const std::vector<float> true_res_y = calculate_residual(
                        closest_hits[i_station],
                        extrapolation_y[i_station],
                        muon_hits_events[0].y,
                        muon_hits_events[0].dy,
                        multiple_scattering_error
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        Catch::Matchers::WithinAbs(true_res_y[0], eps) ||
                        Catch::Matchers::WithinAbs(true_res_y[1], eps) ||
                        Catch::Matchers::WithinAbs(true_res_y[2], eps)
                    );
                }
            }
        }

        WHEN( "Track mathes hit and parallel to the axis OZ (x=10, y=-10, z=0, dx=0, dy=0)" ) {
            MiniState track = MiniState(10, -10, 0, 0, 0);
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
                "Extrapolation of track:  \n"
                "\t station 1 - ( 10,-10) \n"
                "\t station 2 - ( 10,-10) \n"
                "\t station 3 - ( 10,-10) \n"
                "\t station 4 - ( 10,-10) \n"
                "Closest hits: \n"
                "\t station 1 - ( 10,-10), index = 8  \n"
                "\t station 2 - ( 10,-10), index = 32 \n"
                "\t station 3 - ( 10,-10), index = 56 \n"
                "\t station 4 - ( 10,-10), index = 80 \n"
                "Traveled distance: \n"
                "\t station 1 -   0 \n"
                "\t station 2 - 100 \n"
                "\t station 3 - 200 \n"
                "\t station 4 - 300 \n"
            ) {
                const std::vector<int> closest_hits = {8, 32, 56, 80};
                const std::vector<float> trav_dist = {0, 100, 200, 300};
                const std::vector<float> extrapolation_x = { 10,  10,  10,  10};
                const std::vector<float> extrapolation_y = {-10, -10, -10, -10};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station], 
                        Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }

        WHEN( "Track is far away from all hits and parallel to the axis OZ (x=4000, y=4, z=0, dx=0, dy=0)" ) {
            MiniState track = MiniState(4000, 4, 0, 0, 0);
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
                "Extrapolation of track:  \n"
                "\t station 1 - (4000, 4) \n"
                "\t station 2 - (4000, 4) \n"
                "\t station 3 - (4000, 4) \n"
                "\t station 4 - (4000, 4) \n"
                "Closest hits: \n"
                "\t station 1 - ( 20, 0), index = 13  \n"
                "\t station 2 - ( 20, 0), index = 37 \n"
                "\t station 3 - ( 20, 0), index = 61 \n"
                "\t station 4 - ( 20, 0), index = 85 \n"
                "Traveled distance: \n"
                "\t station 1 -   0 \n"
                "\t station 2 - 100 \n"
                "\t station 3 - 200 \n"
                "\t station 4 - 300 \n"
            ) {
                const std::vector<int> closest_hits = {13, 37, 61, 85};
                const std::vector<float> trav_dist = {0, 100, 200, 300};
                const std::vector<float> extrapolation_x = {4000, 4000, 4000, 4000};
                const std::vector<float> extrapolation_y = {4, 4, 4, 4};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(host_features[offset::RES_X + i_station], 
                        Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(host_features[offset::RES_Y + i_station],
                        Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }
        free(host_features);
        cudaFree(dev_muon_hits);
    }
    dev_free_memory();
}

SCENARIO( "General case" ) {
    dev_allocate_memory();

    GIVEN( 
        "Grid of mock hits\n" 
        "There is 24 hits on each station with coordinates x,y: \n"
        "\t (-20, 20) - (-10, 20) - ( 0, 20) - ( 10, 20) - ( 20, 20)\n"
        "\t (-20, 10) - (-10, 10) - ( 0, 10) - ( 10, 10) - ( 20, 10)\n"
        "\t (-20,  0) - (-10,  0) -          - ( 10,  0) - ( 20,  0)\n"
        "\t (-20,-10) - (-10,-10) - ( 0,-10) - ( 10,-10) - ( 20,-10)\n"
        "\t (-20,-20) - (-10,-20) - ( 0,-20) - ( 10,-20) - ( 20,-20)\n"
        "z = 100, 200, 300, 400 for stations 1, 2, 3, 4 respectively\n"
        "Hits indices for the first station: \n"
        "\t 19	-   20	  -   21    -   22    -   23\n"
        "\t 14	-   15	  -   16    -   17    -   18\n"
        "\t 10	-   11	  -         -   12    -   13\n"
        "\t 5	-    6	  -    7    -    8    -    9\n"
        "\t 0	-    1	  -    2    -    3    -    4\n"
        "and so on \n"
        "Let dx = index, dy = index / 2, dz = 0\n"
    ) {
        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = construct_mock_muon_hit();
        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        float *host_features = (float*)malloc(1 * n_features * sizeof(float));

        WHEN( "Track is (x=-3, y=-2, z=0, dx=0.05, dy=0.05)" ) {
            MiniState track = MiniState(-3, -2, 0, 0.05, 0.05);
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
                "\t station 1 - ( 2, 3) \n"
                "\t station 2 - ( 7, 8) \n"
                "\t station 3 - (12,13) \n"
                "\t station 4 - (17,18) \n"
                "Closest hits: \n"
                "\t station 1 - ( 0, 10), index = 16 \n"
                "\t station 2 - (10, 10), index = 41 \n"
                "\t station 3 - (10, 10), index = 65 \n"
                "\t station 4 - (20, 20), index = 95 \n"
                "Traveled distance: \n"
                "\t station 1 - 0   \n"
                "\t station 2 - sqrt(10050) \n"
                "\t station 3 - sqrt(40200) \n"
                "\t station 4 - sqrt(90500) \n"
            ) {
                const std::vector<int> closest_hits = {16, 41, 65, 95};
                const std::vector<float> extrapolation_x = {2, 7, 12, 17};
                const std::vector<float> extrapolation_y = {3, 8, 13, 18};
                const std::vector<float> trav_dist = {0, sqrtf(10050), sqrtf(40200), sqrtf(90500)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }

        WHEN( "Track is (x=-3, y=4, z=0, dx=0.01, dy=-0.05)" ) {
            MiniState track = MiniState(-3, 4, 0, 0.01, -0.05);
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
                "Extrapolation of track:  \n"
                "\t station 1 - ( -2, -1) \n"
                "\t station 2 - ( -1, -6) \n"
                "\t station 3 - (  0,-11) \n"
                "\t station 4 - (  1,-16) \n"
                "Closest hits: \n"
                "\t station 1 - (-10,  0), index = 11 \n"
                "\t station 2 - (  0,-10), index = 31 \n"
                "\t station 3 - (  0,-10), index = 55 \n"
                "\t station 4 - (  0,-20), index = 74 \n"
                "Traveled distance: \n"
                "\t station 1 - 0   \n"
                "\t station 2 - sqrt(10026) \n"
                "\t station 3 - sqrt(40104) \n"
                "\t station 4 - sqrt(90234) \n"
            ) {
                const std::vector<int> closest_hits = {11, 31, 55, 74};
                const std::vector<float> extrapolation_x = {-2, -1, 0, 1};
                const std::vector<float> extrapolation_y = {-1, -6, -11, -16};
                const std::vector<float> trav_dist = {0, sqrtf(10026), sqrtf(40104), sqrtf(90234)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }

        WHEN( "Track is far away from hits (x=1960, y=-27, z=0, dx=0.1, dy=0.1)" ) {
            MiniState track = MiniState(1960, -27, 0, 0.1, 0.1);
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
                "\t station 1 - (1970,-17) \n"
                "\t station 2 - (1980, -7) \n"
                "\t station 3 - (1990,  3) \n"
                "\t station 4 - (2000, 13) \n"
                "Closest hits: \n"
                "\t station 1 - ( 20,-20), index = 4  \n"
                "\t station 2 - ( 20,-10), index = 33 \n"
                "\t station 3 - ( 20,  0), index = 61 \n"
                "\t station 4 - ( 20, 10), index = 90 \n"
                "Traveled distance: \n"
                "\t station 1 - 0   \n"
                "\t station 2 - sqrt(10200) \n"
                "\t station 3 - sqrt(40800) \n"
                "\t station 4 - sqrt(91800) \n"
            ) {
                const std::vector<int> closest_hits = {4, 33, 61, 90};
                const std::vector<float> extrapolation_x = {1970, 1980, 1990, 2000};
                const std::vector<float> extrapolation_y = {-17, -7, 3, 13};
                const std::vector<float> trav_dist = {0, sqrtf(10200), sqrtf(40800), sqrtf(91800)};
                for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
                    const int closest_idx = closest_hits[i_station];
                    const float multiple_scattering_error = COMMON_FACTOR * trav_dist[i_station] * sqrt(trav_dist[i_station]);
                    CHECK(host_features[offset::DTS + i_station] == muon_hits_events[0].delta_time[closest_idx]);
                    CHECK(host_features[offset::TIMES + i_station] == muon_hits_events[0].time[closest_idx]);
                    CHECK(host_features[offset::CROSS + i_station] + muon_hits_events[0].uncrossed[closest_idx] == 2);
                    CHECK_THAT(
                        host_features[offset::RES_X + i_station], Catch::Matchers::WithinAbs((extrapolation_x[i_station] - muon_hits_events[0].x[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                    CHECK_THAT(
                        host_features[offset::RES_Y + i_station], Catch::Matchers::WithinAbs((extrapolation_y[i_station] - muon_hits_events[0].y[closest_idx]) / 
                        sqrt(closest_idx * closest_idx * INVSQRT3 * INVSQRT3 / 4 + multiple_scattering_error * multiple_scattering_error), eps)
                    );
                }
            }
        }
        free(host_features);
        cudaFree(dev_muon_hits);
    }
    dev_free_memory();
}
