#include "IsMuon.cuh"
#include "ConsolidateSciFi.cuh"
#include "SystemOfUnits.h"

__device__ float elliptical_foi_window(
  const float a, 
  const float b, 
  const float c,
  const float momentum
) {
  return a + b * std::exp(-c * momentum / Gaudi::Units::GeV);
}

__device__ std::pair<float,float> field_of_interest(
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const int station,
  const int region,
  const float momentum
) {
  if (momentum < 1000 * Gaudi::Units::GeV) {
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
  } else {
    return {
      dev_muon_foi->param_a_x[station][region],
      dev_muon_foi->param_a_y[station][region]
    };
  }
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

__global__ void is_muon(
  int* dev_atomics_scifi,
  uint* dev_scifi_track_hit_number,
  float* dev_scifi_qop,
  MiniState* dev_scifi_states,
  uint* dev_scifi_track_ut_indices,
  const Muon::HitsSoA* muon_hits,
  int* dev_muon_track_occupancies,
  bool* dev_is_muon,
  const uint* event_list,
  const Muon::Constants::FieldOfInterest* dev_muon_foi,
  const float* dev_muon_momentum_cuts
) {
  const uint number_of_events = gridDim.x;
  const uint event_id = blockIdx.x;
  const uint station_id = threadIdx.y;
  const uint selected_event_number = event_list[event_id];

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
  const int station_offset = muon_hits[selected_event_number].station_offsets[station_id];
  const int number_of_hits = muon_hits[selected_event_number].number_of_hits_per_station[station_id];
  const float station_z = muon_hits[selected_event_number].z[station_offset];
  
  for (uint track_id = threadIdx.x; track_id < number_of_tracks_event; track_id += blockDim.x) {
    const float momentum = 1 / std::abs(scifi_tracks.qop[track_id]);
    const float extrapolation_x = scifi_tracks.states[track_id].x + scifi_tracks.states[track_id].tx * (station_z - scifi_tracks.states[track_id].z);
    const float extrapolation_y = scifi_tracks.states[track_id].y + scifi_tracks.states[track_id].ty * (station_z - scifi_tracks.states[track_id].z);
    const uint track_offset = (event_offset + track_id) * Muon::Constants::n_stations;
    
    dev_muon_track_occupancies[track_offset + station_id] = 0;
    for (int i_hit = 0; i_hit < number_of_hits; ++i_hit) {
      const int idx = station_offset + i_hit;
      if (is_in_window(
        muon_hits[selected_event_number].x[idx],
        muon_hits[selected_event_number].y[idx],
        muon_hits[selected_event_number].dx[idx],
        muon_hits[selected_event_number].dy[idx],
        dev_muon_foi,
        station_id,
        muon_hits[selected_event_number].region_id[idx],
        momentum,
        extrapolation_x,
        extrapolation_y
      )) {
        dev_muon_track_occupancies[track_offset + station_id] += 1;
      }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
      if (momentum < dev_muon_momentum_cuts[0]) {
        dev_is_muon[event_offset + track_id] = false;
      }
      else if (dev_muon_track_occupancies[track_offset + 0] == 0 || dev_muon_track_occupancies[track_offset + 1] == 0) {
        dev_is_muon[event_offset + track_id] = false;
      }
      else if (momentum < dev_muon_momentum_cuts[1]) {
        dev_is_muon[event_offset + track_id] = true;
      }
      else if (momentum < dev_muon_momentum_cuts[2]) {
        dev_is_muon[event_offset + track_id] =
          (dev_muon_track_occupancies[track_offset + 2] != 0 ) || (dev_muon_track_occupancies[track_offset + 3] != 0 );
      }
      else {
        dev_is_muon[event_offset + track_id] = 
          (dev_muon_track_occupancies[track_offset + 2] != 0 ) && (dev_muon_track_occupancies[track_offset + 3] != 0 );
      }
    }
  }
}
