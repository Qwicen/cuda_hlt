#include "catch.hpp"
#include "IsMuon.cuh"
#include "MuonDefinitions.cuh"
#include "MuonFeaturesExtraction.test.cuh"

/*       ___________________________________________________
*       |                                                  |
*       |                           R4                     |
*       |                                                  |
*       |                                                  |
*       |                                                  |
*       |                                                  |
*       |                                                  |
*       |_________________________                         |
*       |              R3         |                        |
*       |                         |                        |
*       |                         |                        |
*       |____________             |                        |
*       |      R2    |            |                        |
*       |_____       |            |                        |
*       |_ R1 |      |            |                        |
*       . |___|______|____________|________________________|
*
*   Station 1: 
*       X:  (0) - Beam Pipe - (30) - R1 - (60) - R2 - (120) - R3 - (240) - R4 - (480)
*       Y:  (0) - Beam Pipe - (25) - R1 - (50) - R2 - (100) - R3 - (200) - R4 - (400)
*   Station 2: 
*       X:  (0) - Beam Pipe - (32.4) - R1 - (65) - R2 - (130) - R3 - (259) - R4 - (518)
*       Y:  (0) - Beam Pipe - (27)   - R1 - (54) - R2 - (108) - R3 - (216) - R4 - (432)
*   Station 3: 
*       X:  (0) - Beam Pipe - (34.8) - R1 - (70) - R2 - (139) - R3 - (278) - R4 - (556)
*       Y:  (0) - Beam Pipe - (29)   - R1 - (58) - R2 - (116) - R3 - (232) - R4 - (464)
*   Station 4: 
*       X:  (0) - Beam Pipe - (37.1) - R1 - (74) - R2 - (149) - R3 - (297) - R4 - (594)
*       Y:  (0) - Beam Pipe - (30.9) - R1 - (62) - R2 - (124) - R3 - (248) - R4 - (495)
*/

SCENARIO( "Track occupancy is zero at any station" ) {
    dev_allocate_memory();

    GIVEN( 
        "Grid of mock hits\n" 
        "There is 24 hits on each station with coordinates x,y: \n"
        "\t (-60, 60) - (-30, 60) - ( 0, 60) - ( 30, 60) - ( 60, 60) \n"
        "\t (-60, 30) - (-30, 30) - ( 0, 30) - ( 30, 30) - ( 60, 30) \n"
        "\t (-60,  0) - (-30,  0) -          - ( 30,  0) - ( 60,  0) \n"
        "\t (-60,-30) - (-30,-30) - ( 0,-30) - ( 30,-30) - ( 60,-30) \n"
        "\t (-60,-60) - (-30,-60) - ( 0,-60) - ( 30,-60) - ( 60,-60) \n"
        "z = 100, 200, 300, 400 for stations 1, 2, 3, 4 respectively \n"
        "dx = dy = 0.1 for every hit\n"
        "Track (x=40, y=40, z=0, dx=0, dy=0) - in first region at any station\n"
    ) {
        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = construct_mock_muon_hit(30);
        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        // Track initialization
        MiniState track = MiniState(40, 40, 0, 0, 0);
        cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

        int *dev_muon_track_occupancies;
        cudaMalloc(&dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int));
        int* host_muon_track_occupancies = (int*)malloc(Muon::Constants::n_stations * sizeof(int));

        bool *dev_is_muon;
        cudaMalloc(&dev_is_muon, 1 * sizeof(bool));
        bool *host_is_muon = (bool*)malloc(1 * sizeof(bool));

        Muon::Constants::FieldOfInterest host_muon_foi;
        Muon::Constants::FieldOfInterest *dev_muon_foi;
        cudaMalloc(&dev_muon_foi, sizeof(Muon::Constants::FieldOfInterest));
        cudaMemcpy(dev_muon_foi, &host_muon_foi, sizeof(Muon::Constants::FieldOfInterest), cudaMemcpyHostToDevice);

        float *dev_muon_momentum_cuts;
        cudaMalloc(&dev_muon_momentum_cuts, 3 * sizeof(float));
        cudaMemcpy(dev_muon_momentum_cuts, Muon::Constants::momentum_cuts, 3 * sizeof(float), cudaMemcpyHostToDevice);

        WHEN( "Momentum of track is 1 GeV / c" ) {
            std::vector<float> qop = {1.0 / 1000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices, 
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n"
                "Occupancies: no\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 0);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 4 GeV / c" ) {
            std::vector<float> qop = {1.0 / 4000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n"
                "Occupancies: no\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 0);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 7 GeV / c" ) {
            std::vector<float> qop = {1.0 / 7000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n"
                "Occupancies: no\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 0);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 15 GeV / c" ) {
            std::vector<float> qop = {1.0 / 15000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n"
                "Occupancies: no\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 0);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }
        free(host_is_muon);
        cudaFree(dev_is_muon);
        cudaFree(dev_muon_hits);
    }
    dev_free_memory();
}

SCENARIO( "Track occupancy is M1" ) {
    dev_allocate_memory();

    GIVEN( 
        "Grid of mock hits\n" 
        "There is 24 hits on each station with coordinates x,y: \n"
        "\t (-60, 60) - (-30, 60) - ( 0, 60) - ( 30, 60) - ( 60, 60) \n"
        "\t (-60, 30) - (-30, 30) - ( 0, 30) - ( 30, 30) - ( 60, 30) \n"
        "\t (-60,  0) - (-30,  0) -          - ( 30,  0) - ( 60,  0) \n"
        "\t (-60,-30) - (-30,-30) - ( 0,-30) - ( 30,-30) - ( 60,-30) \n"
        "\t (-60,-60) - (-30,-60) - ( 0,-60) - ( 30,-60) - ( 60,-60) \n"
        "z = 100, 200, 300, 400 for stations 1, 2, 3, 4 respectively \n"
        "dx = dy = 0.1 for every hit\n"
        "Track (x=40, y=40, z=0, dx=0, dy=0) - in first region at any station\n"
    ) {
        std::vector<Muon::HitsSoA> muon_hits_events;
        Muon::HitsSoA muon_hits = construct_mock_muon_hit(30);
        // One event
        muon_hits_events.push_back(muon_hits);
        
        Muon::HitsSoA *dev_muon_hits;
        cudaMalloc(&dev_muon_hits, muon_hits_events.size() * sizeof(Muon::HitsSoA));
        cudaMemcpy(dev_muon_hits, muon_hits_events.data(), muon_hits_events.size() * sizeof(Muon::HitsSoA), cudaMemcpyHostToDevice);

        // Track initialization
        MiniState track = MiniState(40, 40, 0, 0, 0);
        cudaMemcpy(dev_track, &track, 1 * sizeof(MiniState), cudaMemcpyHostToDevice);

        int *dev_muon_track_occupancies;
        cudaMalloc(&dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int));
        int* host_muon_track_occupancies = (int*)malloc(Muon::Constants::n_stations * sizeof(int));

        bool *dev_is_muon;
        cudaMalloc(&dev_is_muon, 1 * sizeof(bool));
        bool *host_is_muon = (bool*)malloc(1 * sizeof(bool));

        Muon::Constants::FieldOfInterest host_muon_foi;
        Muon::Constants::FieldOfInterest *dev_muon_foi;
        cudaMalloc(&dev_muon_foi, sizeof(Muon::Constants::FieldOfInterest));
        cudaMemcpy(dev_muon_foi, &host_muon_foi, sizeof(Muon::Constants::FieldOfInterest), cudaMemcpyHostToDevice);

        float *dev_muon_momentum_cuts;
        cudaMalloc(&dev_muon_momentum_cuts, 3 * sizeof(float));
        cudaMemcpy(dev_muon_momentum_cuts, Muon::Constants::momentum_cuts, 3 * sizeof(float), cudaMemcpyHostToDevice);


        WHEN( "Momentum of track is 1 GeV / c" ) {
            std::vector<float> qop = {1.0 / 1000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n" 
                "Occupancies: M1\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 100);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 4 GeV / c" ) {
            std::vector<float> qop = {1.0 / 4000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n" 
                "Occupancies: M1\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 100);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 7 GeV / c" ) {
            std::vector<float> qop = {1.0 / 7000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n" 
                "Occupancies: M1\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 100);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }

        WHEN( "Momentum of track is 15 GeV / c" ) {
            std::vector<float> qop = {1.0 / 15000};
            cudaMemcpy(dev_qop, qop.data(), qop.size() * sizeof(float), cudaMemcpyHostToDevice);
            is_muon<<<dim3(1, 4), 32>>>(dev_atomics_scifi, dev_scifi_track_hit_number, dev_qop, dev_track, dev_scifi_track_ut_indices,
                dev_muon_hits, dev_muon_track_occupancies, dev_is_muon, dev_muon_foi, dev_muon_momentum_cuts);
            cudaMemcpy(host_is_muon, dev_is_muon, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_muon_track_occupancies, dev_muon_track_occupancies, Muon::Constants::n_stations * sizeof(int), cudaMemcpyDeviceToHost);
            THEN( 
                "This track is not Muon. Low occupancy.\n" 
                "Occupancies: M1\n"
            ) {
                CHECK(host_is_muon[0] == false);
                CHECK(host_muon_track_occupancies[0] == 100);
                CHECK(host_muon_track_occupancies[1] == 0);
                CHECK(host_muon_track_occupancies[2] == 0);
                CHECK(host_muon_track_occupancies[3] == 0);
            }
        }
        free(host_is_muon);
        cudaFree(dev_is_muon);
        cudaFree(dev_muon_hits);
    }
    dev_free_memory();
}
