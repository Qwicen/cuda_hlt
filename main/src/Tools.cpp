#include "Tools.h"

bool check_velopix_events(
  const std::vector<char>& events,
  const std::vector<uint>& event_offsets,
  int n_events
) {
  int error_count = 0;
  int n_sps_all_events = 0;
  for ( int i_event = 0; i_event < n_events; ++i_event ) {
    const char* raw_input = events.data() + event_offsets[i_event];

    const char* p = events.data() + event_offsets[i_event];
    uint32_t number_of_raw_banks = *((uint32_t*)p); p += sizeof(uint32_t);
    uint32_t* raw_bank_offset = (uint32_t*) p; p += number_of_raw_banks * sizeof(uint32_t);

    uint32_t sensor =  *((uint32_t*)p);  p += sizeof(uint32_t);
    uint32_t sp_count =  *((uint32_t*)p); p += sizeof(uint32_t);

    const auto raw_event = VeloRawEvent(raw_input);
    int n_sps_event = 0;
    for ( int i_raw_bank = 0; i_raw_bank < raw_event.number_of_raw_banks; i_raw_bank++ ) {
      const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[i_raw_bank]);
      n_sps_event += raw_bank.sp_count;
      if ( i_raw_bank != raw_bank.sensor_index ) {
        error_cout << "at raw bank " << i_raw_bank << ", but index = " << raw_bank.sensor_index << std::endl;
        ++error_count;
      }
      if ( raw_bank.sp_count > 0 ) {
        uint32_t sp_word = raw_bank.sp_word[0];
        uint8_t sp = sp_word & 0xFFU;
        if (0 == sp) { continue; };
        const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
        const uint32_t sp_row = sp_addr & 0x3FU;
        const uint32_t sp_col = (sp_addr >> 6);
        const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
      }
    }
    n_sps_all_events += n_sps_event;
  }

  if (error_count>0) {
    error_cout << error_count << " errors detected." << std::endl;
    return false;
  }
  return true;
}

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
