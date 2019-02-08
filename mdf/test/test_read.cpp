#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <map>

#include "raw_bank.hpp"
#include "read_mdf.hpp"
#include "Tools.h"

namespace {
  using std::cout;
  using std::endl;
  using std::ifstream;
  using std::ios;
  using std::make_pair;
  using std::map;
  using std::string;
  using std::to_string;
  using std::unordered_set;
  using std::vector;
} // namespace

int main(int argc, char* argv[])
{
  if (argc <= 1) {
    cout << "usage: test_read <file.mdf>" << endl;
    return -1;
  }

  string filename = {argv[1]};

  // Some storage for reading the events into
  LHCb::MDFHeader header;
  vector<char> read_buffer;
  read_buffer.reserve(1024 * 1024);
  bool eof = false, error = false;

  gsl::span<const char> bank_span;

  ifstream input {filename.c_str(), std::ios::binary};
  if (!input.good()) {
    cout << "failed to open file " << filename << endl;
    return -1;
  }

  // Lambda to copy data from the event-local buffer to the global
  // one, while keeping the global buffer's size consistent with its
  // content
  auto copy_data = [](unsigned int& event_offset, vector<char>& buf, const void* source, size_t s) {
    size_t n_chars = buf.size();
    for (size_t i = 0; i < s; ++i) {
      buf.emplace_back(0);
    }
    ::memcpy(&buf[n_chars], source, s);
    event_offset += s;
  };

  std::tie(eof, error, bank_span) = MDF::read_event(input, header, read_buffer);
  if (eof || error) {
    return -1;
  }

  vector<uint32_t> bank_data;
  vector<uint32_t> bank_offsets;
  bank_offsets.reserve(100);
  bank_offsets.push_back(0);

  // Put the banks in the event-local buffers
  const auto* bank = bank_span.begin();
  const auto* end = bank_span.end();
  while (bank < end) {
    const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);
    if (b->magic() != LHCb::RawBank::MagicPattern) {
      cout << "magic pattern failed: " << std::hex << b->magic() << std::dec << endl;
    }

    // Check if cuda_hlt even knows about this type of bank
    auto cuda_type_it = LHCbToGPU::bank_types.find(b->type());
    if (cuda_type_it == LHCbToGPU::bank_types.end()) {
      bank += b->totalSize();
      continue;
    }

    // Check if we want this bank
    if (cuda_type_it->second != BankTypes::VP) {
      bank += b->totalSize();
      continue;
    }

    auto offset = bank_offsets.back() / sizeof(uint32_t);

    // Store this bank in the event-local buffers
    const uint32_t sourceID = static_cast<uint32_t>(b->sourceID());
    bank_data.push_back(sourceID);
    offset++;

    auto b_start = b->begin<uint32_t>();
    auto b_end = b->end<uint32_t>();

    while (b_start != b_end) {
      const uint32_t raw_data = *b_start;
      bank_data.emplace_back(raw_data);

      b_start++;
      offset++;
    }

    // Record raw bank offset
    bank_offsets.push_back(offset * sizeof(uint32_t));

    // Move to next raw bank
    bank += b->totalSize();
  }

  unsigned int event_offset = 0;
  uint32_t n_banks = bank_offsets.size() - 1;
  vector<char> buf;
  cout << "n_banks: " << n_banks << endl;
  copy_data(event_offset, buf, &n_banks, sizeof(n_banks));

  // Copy in bank offsets
  copy_data(event_offset, buf, bank_offsets.data(), bank_offsets.size() * sizeof(uint32_t));

  // Copy in bank data
  copy_data(event_offset, buf, bank_data.data(), bank_data.size() * sizeof(uint32_t));

  std::vector<unsigned int> event_offsets = {0, event_offset};
  cout << event_offset << endl;
  check_velopix_events(buf, event_offsets, 1);
}
