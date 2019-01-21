#include "Tools.h"

/**
 * @brief Obtains results statistics.
 */
std::map<std::string, float> calcResults(std::vector<float>& times){
    // sqrt ( E( (X - m)2) )
    std::map<std::string, float> results;
    float deviation = 0.0f, variance = 0.0f, mean = 0.0f, min = FLT_MAX, max = 0.0f;

    for(auto it = times.begin(); it != times.end(); it++){
        const float seconds = (*it);
        mean += seconds;
        variance += seconds * seconds;

        if (seconds < min) min = seconds;
        if (seconds > max) max = seconds;
    }

    mean /= times.size();
    variance = (variance / times.size()) - (mean * mean);
    deviation = std::sqrt(variance);

    results["variance"] = variance;
    results["deviation"] = deviation;
    results["mean"] = mean;
    results["min"] = min;
    results["max"] = max;

    return results;
}

/**
 * @brief Prints the memory consumption of the device.
 */
void print_gpu_memory_consumption() {
  size_t free_byte;
  size_t total_byte;
  cudaCheck(cudaMemGetInfo(&free_byte, &total_byte));
  float free_percent = (float)free_byte / total_byte * 100;
  float used_percent = (float)(total_byte - free_byte) / total_byte * 100;
  verbose_cout << "GPU memory: " << free_percent << " percent free, "
    << used_percent << " percent used " << std::endl;
}

std::pair<size_t, std::string> set_device(int cuda_device) {
  int n_devices = 0;
  cudaDeviceProp device_properties;
  cudaCheck(cudaGetDeviceCount(&n_devices));

  debug_cout << "There are " << n_devices << " CUDA devices available" << std::endl;
  for (int cd = 0; cd < n_devices; ++cd) {
     cudaDeviceProp device_properties;
     cudaCheck(cudaGetDeviceProperties(&device_properties, cd));
     debug_cout << std::setw(3) << cd << " " << device_properties.name << std::endl;
  }

  if (cuda_device >= n_devices) {
     error_cout << "Chosen device (" << cuda_device << ") is not available." << std::endl;
     return {0, ""};
  }
  debug_cout << std::endl;

  cudaCheck(cudaSetDevice(cuda_device));
  cudaCheck(cudaGetDeviceProperties(&device_properties, cuda_device));
  return {n_devices, device_properties.name};
}

void read_muon_events_into_arrays(
  Muon::HitsSoA *muon_station_hits,
  const char* events,
  const uint* event_offsets,
  const int n_events
) {
  for (int i_event = 0; i_event < n_events; ++i_event) {
    const char* raw_input = events + event_offsets[i_event];
    std::copy_n((int*) raw_input, Muon::Constants::n_stations, muon_station_hits[i_event].number_of_hits_per_station);
    raw_input += sizeof(int) * Muon::Constants::n_stations;
    muon_station_hits[i_event].station_offsets[0] = 0;
    for(int i_station = 1; i_station < Muon::Constants::n_stations; ++i_station) {
      muon_station_hits[i_event].station_offsets[i_station] = muon_station_hits[i_event].station_offsets[i_station - 1]
                                                            + muon_station_hits[i_event].number_of_hits_per_station[i_station - 1];
    }
    for(int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
      const int station_offset = muon_station_hits[i_event].station_offsets[i_station];
      const int number_of_hits = muon_station_hits[i_event].number_of_hits_per_station[i_station];
      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].tile[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].x[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].dx[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].y[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].dy[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].z[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((float*) raw_input, number_of_hits, &( muon_station_hits[i_event].dz[station_offset]) );
      raw_input += sizeof(float) * number_of_hits;
      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].uncrossed[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;
      std::copy_n((unsigned int*) raw_input, number_of_hits, &( muon_station_hits[i_event].time[station_offset]) );
      raw_input += sizeof(unsigned int) * number_of_hits;
      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].delta_time[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;
      std::copy_n((int*) raw_input, number_of_hits, &( muon_station_hits[i_event].cluster_size[station_offset]) );
      raw_input += sizeof(int) * number_of_hits;
    }
  }
}
 void check_muon_events(
   const Muon::HitsSoA * muon_station_hits,
   const int n_output_hits_per_event,
   const int n_events
) {
  int total_number_of_hits = 0;
  for (int i_event = 0; i_event < n_events; ++i_event) {
    float number_of_hits_per_event = 0;
    for (int i_station = 0; i_station < Muon::Constants::n_stations; ++i_station) {
      const int station_offset = muon_station_hits[i_event].station_offsets[i_station];
      const int number_of_hits = muon_station_hits[i_event].number_of_hits_per_station[i_station];
      number_of_hits_per_event += number_of_hits;
      debug_cout << "checks on station " << i_station << ", with " << number_of_hits << " hits" << std::endl;
      for (int i_hit = 0; i_hit < n_output_hits_per_event; ++i_hit) {
	debug_cout << "\t at hit " << i_hit << ", " <<
          "tile = "         << muon_station_hits->tile[station_offset + i_hit]         << ", " <<
          "x = "            << muon_station_hits->x[station_offset + i_hit]            << ", " <<
          "dx = "           << muon_station_hits->dx[station_offset + i_hit]           << ", " <<
          "y = "            << muon_station_hits->y[station_offset + i_hit]            << ", " <<
          "dy = "           << muon_station_hits->dy[station_offset + i_hit]           << ", " <<
          "z = "            << muon_station_hits->z[station_offset + i_hit]            << ", " <<
          "dz = "           << muon_station_hits->dz[station_offset + i_hit]           << ", " <<
          "uncrossed = "    << muon_station_hits->uncrossed[station_offset + i_hit]    << ", " <<
          "time = "         << muon_station_hits->time[station_offset + i_hit]         << ", " <<
          "delta_time = "   << muon_station_hits->delta_time[station_offset + i_hit]   << ", " <<
          "cluster_size = " << muon_station_hits->cluster_size[station_offset + i_hit] << ", " <<
	  std::endl;
      }
    }
    total_number_of_hits += number_of_hits_per_event;
    debug_cout << "# of Muon hits = " << number_of_hits_per_event << std::endl;
  }
  debug_cout << "average # of Muon hits / event = " << (float) total_number_of_hits / n_events<< std::endl;
}

void read_muon_coords(Muon::MuonCoords *muon_coords, const char *raw_input) {
  std::copy_n((int *) raw_input, Muon::Constants::n_stations, muon_coords->number_of_hits_per_station);
  raw_input += sizeof(int) * Muon::Constants::n_stations;
  std::partial_sum(
      muon_coords->number_of_hits_per_station,
      muon_coords->number_of_hits_per_station + Muon::Constants::n_stations - 1,
      muon_coords->station_offsets + 1
  );
  muon_coords->digit_tile_station_offsets[0] = 0;
  for (uint station = 0; station < Muon::Constants::n_stations; station++) {
    const int station_offset = muon_coords->station_offsets[station];
    const int number_of_hits = muon_coords->number_of_hits_per_station[station];
    const int digit_tile_station_offset = muon_coords->digit_tile_station_offsets[station];
    std::copy_n((int *) raw_input, number_of_hits, muon_coords->uncrossed + station_offset);
    raw_input += sizeof(int) * number_of_hits;
    const int number_of_uncrossed = std::accumulate(
        muon_coords->uncrossed + station_offset,
        muon_coords->uncrossed + station_offset + number_of_hits,
        0
    );
    const int number_of_digit_tiles = number_of_hits * 2 - number_of_uncrossed;
    muon_coords->number_of_digit_tiles_per_station[station] = number_of_digit_tiles;
    if (station + 1 < Muon::Constants::n_stations) {
      muon_coords->digit_tile_station_offsets[station + 1] = digit_tile_station_offset + number_of_digit_tiles;
    }
    std::copy_n((int *) raw_input, number_of_digit_tiles, muon_coords->digit_tile + digit_tile_station_offset);
    raw_input += sizeof(int) * number_of_digit_tiles;
    std::copy_n((uint *) raw_input, number_of_hits, muon_coords->digitTDC1 + station_offset);
    raw_input += sizeof(uint) * number_of_hits;
    std::copy_n((uint *) raw_input, number_of_hits, muon_coords->digitTDC2 + station_offset);
    raw_input += sizeof(uint) * number_of_hits;
  }
}

void read_muon_coords_folder(
    Muon::MuonCoords *muon_coords,
    const char *events,
    const uint *event_offsets,
    const int n_events
) {
  for (int event = 0; event < n_events; event++) {
    const char *raw_input = events + event_offsets[event];
    read_muon_coords(&muon_coords[event], raw_input);
  }
}

void check_muon_coords(Muon::MuonCoords *muon_coords, const int n_output_hits) {
  for (uint station = 0; station < Muon::Constants::n_stations; ++station) {
    const int station_offset = muon_coords->station_offsets[station];
    const int number_of_hits = muon_coords->number_of_hits_per_station[station];
    const int digit_tile_station_offset = muon_coords->digit_tile_station_offsets[station];
    const int number_of_digit_tiles = muon_coords->number_of_digit_tiles_per_station[station];
    debug_cout << "checks on station " << station << ", with " << number_of_hits << " hits" << std::endl;
    int digit_tile_number = 0;
    for (int hit = 0; hit < std::min(number_of_hits, n_output_hits); ++hit) {
      const int uncrossed = muon_coords->uncrossed[station_offset + hit];
      debug_cout << "\t at hit " << hit << ", " << "uncrossed = " << uncrossed << ", ";
      if (uncrossed == 0) {
        debug_cout << "digit_tiles = (" << (muon_coords->digit_tile[digit_tile_station_offset + digit_tile_number])
                   << ", " << (muon_coords->digit_tile[digit_tile_station_offset + digit_tile_number + 1]) << "), ";
        digit_tile_number += 2;
      } else {
        debug_cout << "digit_tile = " << (muon_coords->digit_tile[digit_tile_station_offset + digit_tile_number])
                   << ", ";
        digit_tile_number++;
      }
      debug_cout << "digitTDC1 = " << (muon_coords->digitTDC1[station_offset + hit]) << ", "
                 << "digitTDC2 = " << (muon_coords->digitTDC2[station_offset + hit]) << std::endl;
    }
  }
}

void check_muon_coords_folder(Muon::MuonCoords *muon_coords, const int n_output_hits, const int n_events) {
  for (int event = 0; event < n_events; event++) {
    check_muon_coords(&muon_coords[event], n_output_hits);
  }
}
