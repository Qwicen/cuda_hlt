#include <vector>
#include <string>

#include "MDFReader.h"

#include "read_mdf.hpp"
#include "odin.hpp"

namespace {
  using std::pair;
  using std::string;
  using std::vector;
} // namespace

void MDFReader::read_events(uint number_of_events_requested, uint start_event_offset)
{

  size_t n_read = 0;
  LHCbToGPU::buffer_map buffers;
  vector<LHCb::ODIN> odins;

  auto bank_type = *begin(types());
  auto foldername = folder(bank_type);
  auto filenames = list_folder(foldername, "mdf");
  vector<string> files;
  files.reserve(filenames.size());
  for (auto name : filenames) {
    files.emplace_back(foldername + "/" + name);
  }
  std::tie(n_read, buffers, odins) = MDF::read_events(number_of_events_requested, files, types(), start_event_offset);

  for (auto bank_type : types()) {
    auto it = buffers.find(bank_type);
    if (it == end(buffers)) {
      throw StrException(string {"Cannot find buffer for bank type "} + bank_name(bank_type));
    }
    auto& entry = it->second;
    ;
    check_events(bank_type, entry.first, entry.second, number_of_events_requested);
  }

  // TODO Remove: Temporal check to understand if number_of_events_requested is the same as number_of_events
  const int number_of_events = begin(buffers)->second.second.size() - 1;
  if (number_of_events_requested != number_of_events) {
    throw StrException("Number of events requested differs from number of events read.");
  }

  for (auto bank_type : types()) {
    auto it = buffers.find(bank_type);
    const auto& ev_buf = it->second.first;
    const auto& offsets_buf = it->second.second;

    // Copy raw data to pinned host memory
    char* events_mem = nullptr;
    uint* offsets_mem = nullptr;
    cudaCheck(cudaMallocHost((void**) &events_mem, ev_buf.size()));
    cudaCheck(cudaMallocHost((void**) &offsets_mem, offsets_buf.size() * sizeof(uint)));
    std::copy_n(std::begin(ev_buf), ev_buf.size(), events_mem);
    std::copy_n(std::begin(offsets_buf), offsets_buf.size(), offsets_mem);

    add_events(bank_type, {events_mem, ev_buf.size()}, {offsets_mem, offsets_buf.size()});
  }
}
