#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>

#include "mdf_header.hpp"
#include "read_mdf.hpp"
#include "raw_bank.hpp"
#include "raw_helpers.hpp"

namespace {
  using gsl::span;
  using std::array;
  using std::cerr;
  using std::cout;
  using std::endl;
  using std::ifstream;
  using std::make_tuple;
  using std::vector;

  const bool ignoreChecksum = false;
} // namespace

std::tuple<size_t, LHCbToGPU::buffer_map, std::vector<LHCb::ODIN>> MDF::read_events(
  size_t n,
  const std::vector<std::string>& files,
  const std::unordered_set<BankTypes>& types,
  size_t start_event)
{

  LHCbToGPU::buffer_map buffers;
  for (auto bank_type : types) {
    auto r = buffers.emplace(bank_type, make_pair(vector<char> {}, vector<unsigned int> {}));
    // Reserve some memory
    auto& banks = r.first->second.first;
    banks.reserve(n * 100 * 1024);
    // Reserve for 100 banks per event, more than enough
    auto& offsets = r.first->second.second;
    offsets.reserve(n * 100);
    offsets.push_back(0);
  }

  vector<LHCb::ODIN> odins;
  odins.reserve(n);

  // Some storage for reading the events into
  LHCb::MDFHeader header;
  vector<char> buffer;
  buffer.reserve(1024 * 1024);
  bool eof = false, error = false;

  gsl::span<const char> bank_span;
  size_t n_read = 0;
  unsigned int event_size = 0;

  array<std::vector<uint32_t>, NBankTypes> bank_offsets;
  array<std::vector<uint32_t>, NBankTypes> bank_datas;

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

  for (const auto& filename : files) {
    ifstream input {filename.c_str(), std::ios::binary};
    if (!input.good()) {
      cout << "failed to open file " << filename << endl;
      break;
    }

    while (n_read++ < n) {

      std::tie(eof, error, bank_span) = read_event(input, header, buffer);
      if (eof || error) {
        break;
      }

      // Skip some events
      if (n_read < start_event) {
        continue;
      }

      // Clear bank offsets
      for (auto& bo : bank_offsets) {
        bo.clear();
        bo.push_back(0);
      }

      // Clear bank data
      for (auto& bd : bank_datas) {
        bd.clear();
      }

      // Put the banks in the event-local buffers
      const auto* bank = bank_span.begin();
      const auto* end = bank_span.end();
      while (bank < end) {
        const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);
        if (b->magic() != LHCb::RawBank::MagicPattern) {
          cout << "magic pattern failed: " << std::hex << b->magic() << std::dec << endl;
        }

        // Decode the odin bank
        if (b->type() == LHCb::RawBank::ODIN) {
          odins.emplace_back(decode_odin(b));
        }

        // Check if cuda_hlt even knows about this type of bank
        auto cuda_type_it = LHCbToGPU::bank_types.find(b->type());
        if (cuda_type_it == LHCbToGPU::bank_types.end()) {
          bank += b->totalSize();
          continue;
        }

        // Check if we want this bank
        auto buf_it = buffers.find(cuda_type_it->second);
        if (buf_it == buffers.end()) {
          bank += b->totalSize();
          continue;
        }

        auto index = to_integral(cuda_type_it->second);
        auto& bank_data = bank_datas[index];
        auto& offsets = bank_offsets[index];
        auto offset = offsets.back() / sizeof(uint32_t);

        // Store this bank in the event-local buffers
        bank_data.push_back(b->sourceID());
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
        offsets.push_back(offset * sizeof(uint32_t));

        // Move to next raw bank
        bank += b->totalSize();
      }

      // Fill the output buffers from the event-local buffers in the right format:
      // - number of raw banks (uint32_t)
      // - raw bank offset within event as number of char (one more than number of banks)
      // - raw bank data as uint32_t
      for (auto& entry : buffers) {
        auto& buf = entry.second.first;

        auto index = to_integral(entry.first);
        auto& event_offsets = entry.second.second;
        auto event_offset = event_offsets.back();

        // Copy in number of banks
        const auto& offsets = bank_offsets[index];
        uint32_t n_banks = offsets.size() - 1;
        copy_data(event_offset, buf, &n_banks, sizeof(n_banks));

        // Copy in bank offsets
        copy_data(event_offset, buf, offsets.data(), offsets.size() * sizeof(uint32_t));

        // Copy in bank data
        const auto& bank_data = bank_datas[index];
        copy_data(event_offset, buf, bank_data.data(), bank_data.size() * sizeof(uint32_t));

        event_offsets.push_back(event_offset);
      }
    }
  }
  n_read = n_read > 0 ? n_read - 1 : 0;
  return make_tuple(n_read, std::move(buffers), std::move(odins));
}

// return eof, error, span that covers all banks in the event
std::tuple<bool, bool, gsl::span<char>>
MDF::read_event(ifstream& input, LHCb::MDFHeader& h, vector<char>& buffer, bool dbg)
{
  int rawSize = sizeof(LHCb::MDFHeader);
  // Read directly into the header
  input.read(reinterpret_cast<char*>(&h), rawSize);
  if (input.good()) {
    return read_banks(input, h, buffer, dbg);
  }
  else if (input.eof()) {
    cout << "Cannot read more data  (Header). End-of-File reached." << endl;
    return {true, false, {}};
  }
  else {
    cerr << "Failed to read header" << endl;
    return {false, true, {}};
  }
}

// return eof, error, span that covers all banks in the event
std::tuple<bool, bool, gsl::span<char>>
MDF::read_banks(ifstream& input, const LHCb::MDFHeader& h, vector<char>& buffer, bool dbg)
{
  int rawSize = sizeof(LHCb::MDFHeader);
  unsigned int checksum = h.checkSum();
  int compress = h.compression() & 0xF;
  int expand = (h.compression() >> 4) + 1;
  int hdrSize = h.subheaderLength();
  int readSize = h.recordSize() - rawSize;
  int chkSize = h.recordSize() - 4 * sizeof(int);
  int alloc_len =
    (rawSize + readSize + sizeof(LHCb::MDFHeader) + sizeof(LHCb::RawBank) + sizeof(int) +
     (compress ? expand * readSize : 0));

  // Build the DAQ status bank that contains the header
  auto build_bank = [rawSize, &h](char* address) {
    auto* b = reinterpret_cast<LHCb::RawBank*>(address);
    b->setMagic();
    b->setType(LHCb::RawBank::DAQ);
    // Reverse engineered, somehow...
    b->setSize(rawSize + 1);
    b->setVersion(DAQ_STATUS_BANK);
    b->setSourceID(0);
    ::memcpy(b->data(), &h, rawSize);
    return b;
  };

  if (dbg) {
    cout << "Size: " << std::setw(6) << h.recordSize() << " Compression:" << compress << " Checksum: 0x" << std::hex
         << checksum << std::dec << endl;
  }

  // accomodate for potential padding of MDF header bank!
  buffer.reserve(alloc_len + sizeof(int) + sizeof(LHCb::RawBank));

  auto* b = build_bank(&buffer[0]);
  int bnkSize = b->totalSize();
  char* bptr = (char*) b->data();
  auto* hdr = reinterpret_cast<LHCb::MDFHeader*>(bptr);
  if (compress != 0) {
    std::vector<char> tmp;
    tmp.reserve(readSize + rawSize);
    ::memcpy(tmp.data(), &h, rawSize); // Need to copy header to get checksum right

    input.read(tmp.data() + rawSize, readSize);
    if (input.eof()) {
      cout << "Cannot read more data  (Header). End-of-File reached." << endl;
      return {true, false, {}};
    }
    else if (input.fail()) {
      cerr << "Failed to read banks" << endl;
      return {false, true, {}};
    }

    int space_retry = 0;
    while (space_retry++ < 5) {
      if (space_retry > 1) {
        alloc_len *= 2;
        if (dbg) {
          cout << "Retry with increased buffer space of " << alloc_len << " bytes." << endl;
        }
        buffer.reserve(alloc_len + sizeof(int) + sizeof(LHCb::RawBank));
        // Rebuild the DAQ status bank on reallocate
        b = build_bank(&buffer[0]);
        bnkSize = b->totalSize();
        bptr = reinterpret_cast<char*>(b->data());
        hdr = reinterpret_cast<LHCb::MDFHeader*>(bptr);
      }

      // Checksum if requested
      if (ignoreChecksum) {
        hdr->setChecksum(0);
      }
      else if (checksum) {
        auto c = LHCb::genChecksum(1, &tmp[0] + 4 * sizeof(int), chkSize);
        if (checksum != c) {
          cerr << "Checksum doesn't match: " << std::hex << c << " instead of 0x" << checksum << std::dec << endl;
          return {false, true, {}};
        }
      }

      // Checksum is correct...from all we know data integrity is proven
      size_t new_len = 0;

      // NOTE: Figured out the magic +1 from dumping stuff until it
      // made sense...
      auto* src = reinterpret_cast<unsigned char*>(&tmp[0]) + rawSize + 1;
      auto* ptr = reinterpret_cast<unsigned char*>(&buffer[0]) + bnkSize;
      size_t space_size = buffer.capacity() - bnkSize;
      // NOTE: Figured out the magic h.size() + hdrSize - 1 from
      // dumping stuff until it made sense...
      if (LHCb::decompressBuffer(compress, ptr, space_size, src, h.size() + hdrSize - 1, new_len)) {
        hdr->setSize(new_len);
        hdr->setCompression(0);
        hdr->setChecksum(0);
        return {false, false, {buffer.data(), bnkSize + new_len}};
      }
    }
    cerr << "Failed to read compressed data." << endl;
    return {false, true, {}};
  }
  else {
    // Read uncompressed data file...
    input.read(bptr + rawSize, readSize);
    if (input.eof()) {
      cout << "Cannot read more data  (Header). End-of-File reached." << endl;
      return {true, false, {}};
    }
    else if (input.fail()) {
      cerr << "Failed to read banks" << endl;
      return {false, true, {}};
    }

    if (ignoreChecksum) {
      hdr->setChecksum(0);
    }
    else if (checksum) {
      auto c = LHCb::genChecksum(1, bptr + 4 * sizeof(int), chkSize);
      if (checksum != c) {
        cerr << "Checksum doesn't match: 0x" << std::hex << c << " instead of 0x" << checksum << std::dec << endl;
        return {false, true, {}};
      }
    }
    return {false, false, {buffer.data(), bnkSize + h.size() - 1}};
  }
}

// Decode the ODIN bank
LHCb::ODIN MDF::decode_odin(const LHCb::RawBank* bank)
{
  LHCb::ODIN odin;
  unsigned long long temp64 {0};
  unsigned int temp32 {0};
  const unsigned int* odinData = bank->data();

  // Fill the ODIN object
  odin.version = bank->version();
  odin.run_number = odinData[LHCb::ODIN::Data::RunNumber];
  odin.orbit_number = odinData[LHCb::ODIN::Data::OrbitNumber];

  temp64 = odinData[LHCb::ODIN::Data::L0EventIDHi];
  odin.event_number = (temp64 << 32) + odinData[LHCb::ODIN::Data::L0EventIDLo];

  temp32 = odinData[LHCb::ODIN::Data::EventType];
  odin.event_type =
    (temp32 & LHCb::ODIN::EventTypeMasks::EventTypeMask) >> LHCb::ODIN::EventTypeBitsEnum::EventTypeBits;
  odin.calibration_step =
    (temp32 & LHCb::ODIN::EventTypeMasks::CalibrationStepMask) >> LHCb::ODIN::EventTypeBitsEnum::CalibrationStepBits;

  odin.tck = odinData[LHCb::ODIN::Data::TriggerConfigurationKey];
  return odin;
}

void MDF::dump_hex(const char* start, int size)
{
  const auto* content = start;
  size_t m = 0;
  cout << std::hex << std::setw(7) << m << " ";
  auto prev = cout.fill();
  auto flags = cout.flags();
  while (content < start + size) {
    if (m % 32 == 0 && m != 0) {
      cout << endl;
      cout << std::setw(7) << m << " ";
    }
    cout << std::setw(2) << std::setfill('0') << ((int) (*content) & 0xff);
    ++m;
    if (m % 2 == 0) {
      cout << " ";
    }
    ++content;
  }
  cout << std::dec << std::setfill(prev) << endl;
  cout.setf(flags);
}
