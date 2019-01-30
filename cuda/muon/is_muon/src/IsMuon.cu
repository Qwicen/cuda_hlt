#include "IsMuon.cuh"
#include "ConsolidateSciFi.cuh"

__global__ void is_muon(
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_scifi_track_ut_indices,
  const Muon::HitsSoA* muon_hits,
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

  const uint number_of_tracks_event = scifi_tracks.number_of_tracks(event_id);
  const uint event_offset = scifi_tracks.tracks_offset(event_id);
  for (uint track_id = threadIdx.x; track_id < number_of_tracks_event; track_id += blockDim.x) {
    const int station_offset = muon_hits[event_id].station_offsets[station_id];
    const int number_of_hits = muon_hits[event_id].number_of_hits_per_station[station_id];
    const float station_z = muon_hits[event_id].z[station_offset];
    const float station_z0 = muon_hits[event_id].z[muon_hits[event_id].station_offsets[0]];

    const float extrapolation_x = scifi_tracks.states[track_id].x + scifi_tracks.states[track_id].tx * (station_z - scifi_tracks.states[track_id].z);
    const float extrapolation_y = scifi_tracks.states[track_id].y + scifi_tracks.states[track_id].ty * (station_z - scifi_tracks.states[track_id].z);
    const float extrapolation_x0 = scifi_tracks.states[track_id].x + scifi_tracks.states[track_id].tx * (station_z0 - scifi_tracks.states[track_id].z);
    const float extrapolation_y0 = scifi_tracks.states[track_id].y + scifi_tracks.states[track_id].ty * (station_z0 - scifi_tracks.states[track_id].z);

    bool occupancy = false;
    for (int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;
      
      
    }
  }
  
  dev_is_muon[0] = true;
}

__device__ float elliptical_foi_window(
  const float a, 
  const float b, 
  const float c,
  const float momentum
) {
  const int to_gev = 1e-9;
  return a + b * std::exp(-c * momentum * to_gev);
}

__device__ std::pair<float,float> field_of_interest(
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const int station,
  const int region,
  const float momentum
) {
  const float momentum_threshold = 1e6; 
  if (momentum < momentum_threshold) {
    
    return {
      elliptical_foi_window(
        dev_muon_foi->param_a_x[station][region], 
        dev_muon_foi->param_b_x[station][region], 
        dev_muon_foi->param_c_x[station][region],
        momentum),
        elliptical_foi_window(
        dev_muon_foi->param_a_y[station][region], 
        dev_muon_foi->param_b_y[station][region], 
        dev_muon_foi->param_c_y[station][region],
        momentum)
      };
  }
  return { dev_muon_foi->param_a_x[station][region], dev_muon_foi->param_a_y[station][region] };
}

__device__ bool is_in_window(
  const float hit_x,
  const float hit_y,
  const float hit_dx,
  const float hit_dy,
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const int station,
  const int region,
  const float momentum,
  const float extrapolation_x,
  const float extrapolation_y
) {
  std::pair<float, float> foi = field_of_interest(dev_muon_foi, station, region, momentum);
  
  return (fabs(hit_x - extrapolation_x) < hit_dx * foi.first * dev_muon_foi->factor) &&
    (fabs(hit_y - extrapolation_y) < hit_dy * foi.second * dev_muon_foi->factor);
}
