#include "ConsolidateUT.cuh"

__global__ void consolidate_ut_tracks(
  uint* dev_ut_hits,
  uint* dev_ut_hit_offsets,
  char* dev_ut_track_hits,
  int* dev_atomics_ut,
  uint* dev_ut_track_hit_number,
  float* dev_ut_qop,
  uint* dev_ut_track_velo_indices,
  const UT::TrackHits* dev_veloUT_tracks,
  const uint* dev_unique_x_sector_layer_offsets)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];
  const uint total_number_of_hits = dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors];
  const UT::TrackHits* event_veloUT_tracks = dev_veloUT_tracks + event_number * UT::Constants::max_num_tracks;
  
  const UT::Hits ut_hits {dev_ut_hits, total_number_of_hits};
  const UT::HitOffsets ut_hit_offsets {dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};
  const auto event_offset = ut_hit_offsets.event_offset();

  // Create consolidated SoAs.
  UT::Consolidated::Tracks ut_tracks {(uint*) dev_atomics_ut,
                                      dev_ut_track_hit_number,
                                      dev_ut_qop,
                                      dev_ut_track_velo_indices,
                                      event_number,
                                      number_of_events};
  const uint number_of_tracks_event = ut_tracks.number_of_tracks(event_number);

  // Loop over tracks.
  for (uint i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {
    ut_tracks.velo_track[i] = event_veloUT_tracks[i].velo_track_index;
    ut_tracks.qop[i] = event_veloUT_tracks[i].qop;
    UT::Consolidated::Hits consolidated_hits = ut_tracks.get_hits(dev_ut_track_hits, i);
    const UT::TrackHits track = event_veloUT_tracks[i];

    // Lambda for populating arrays.
    auto populate = [&track](uint32_t* __restrict__ a, uint32_t* __restrict__ b) {
      int hit_number = 0;
      for (int i = 0; i < UT::Constants::n_layers; ++i) {
        const auto hit_index = track.hits[i];
        if (hit_index != -1) {
          a[hit_number++] = b[hit_index];
        }
      }
    };

    // Populate the plane code.
    auto populate_plane_code = [](uint8_t* __restrict__ a, const UT::TrackHits& track) {
      int hit_number = 0;
      for (uint8_t i = 0; i < UT::Constants::n_layers; ++i) {
        const auto hit_index = track.hits[i];
        if (hit_index != -1) {
          a[hit_number++] = i;
        }
      }
    };
    
    // Populate the consolidated hits.
    populate((uint32_t*) consolidated_hits.yBegin, (uint32_t*) ut_hits.yBegin + event_offset);
    populate((uint32_t*) consolidated_hits.yEnd, (uint32_t*) ut_hits.yEnd + event_offset);
    populate((uint32_t*) consolidated_hits.zAtYEq0, (uint32_t*) ut_hits.zAtYEq0 + event_offset);
    populate((uint32_t*) consolidated_hits.xAtYEq0, (uint32_t*) ut_hits.xAtYEq0 + event_offset);
    populate((uint32_t*) consolidated_hits.LHCbID, (uint32_t*) ut_hits.LHCbID + event_offset);
    populate((uint32_t*) consolidated_hits.weight, (uint32_t*) ut_hits.weight + event_offset);
    populate_plane_code(consolidated_hits.plane_code, track);
  }
}
