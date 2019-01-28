#include "IsMuon.cuh"
#include "ConsolidateSciFi.cuh"

__global__ void is_muon(
    int* dev_atomics_scifi,
    uint* dev_scifi_track_hit_number,
    float* dev_scifi_qop,
    MiniState* dev_scifi_states,
    uint* dev_scifi_track_ut_indices,
    bool* dev_is_muon,
    const Muon::Constants::FieldOfInterest* dev_muon_foi,
    const float* dev_muon_momentum_cuts
) {
    const uint number_of_events = gridDim.x;
    const uint event_id = blockIdx.x;
    const uint station_id = blockIdx.y;

    SciFi::Consolidated::Tracks scifi_tracks {
        (uint*)dev_atomics_scifi,
        dev_scifi_track_hit_number,
        dev_scifi_qop,
        dev_scifi_states,
        dev_scifi_track_ut_indices,
        event_id,
        number_of_events
    };

    dev_is_muon[0] = true;
}
