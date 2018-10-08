#include <list>
#include <algorithm>
#include "Common.h"
#include "Logger.h"

struct MemoryManager {
  size_t max_available_memory = (size_t) 8 * 1024 * 1024 * 1024; // 8 GiB
  constexpr static uint guarantee_alignment = 256;

  /**
   * @brief A memory segment is composed of a start
   *        and size, both referencing bytes.
   *        The tag can either be -1 (free), or i \in {0-n},
   *        which means it is occupied by argument i
   */
  struct MemorySegment {
    uint start;
    size_t size;
    int tag;
  };

  std::list<MemorySegment> memory_segments = {{0, max_available_memory, -1}};
  size_t total_memory_required = 0;

  MemoryManager() = default;

  /**
   * @brief Sets the max_available_memory of this manager.
   *        Note: This triggers a free_all to restore the memory_segments
   *        to a valid state. This operation is very disruptive.
   */
  void set_reserved_memory(size_t reserved_memory) {
    max_available_memory = reserved_memory;
    free_all();
  }

  /**
   * @brief Reserves a memory request of size requested_size.
   *        It can be overriden by other memory managers. The
   *        base version finds the first available segment. If
   *        there are no available segments of the requested size,
   *        it throws an exception.
   */
  uint reserve(int tag, size_t requested_size) {
    // Aligned requested size
    const size_t aligned_request = requested_size + guarantee_alignment - 1
      - ((requested_size + guarantee_alignment - 1) % guarantee_alignment);

    if (logger::ll.verbosityLevel >= 4) {
      debug_cout << "MemoryManager: Requested "
        << requested_size << " B (" << aligned_request << " B aligned)" << std::endl;
    }

    auto it = memory_segments.begin();
    for (; it!=memory_segments.end(); ++it) {
      if (it->tag == -1 && it->size >= aligned_request) {
        break;
      }
    }

    if (it == memory_segments.end()) {
      print();
      throw StrException("Reserve: Requested size could not be met ("
        + std::to_string(((float) aligned_request) / (1024*1024)) + " MiB)");
    }

    // Start of allocation
    uint start = it->start;

    // Update current segment
    it->start += aligned_request;
    it->size -= aligned_request;
    if (it->size == 0) {
      it = memory_segments.erase(it);
    }

    // Insert an occupied segment
    auto segment = MemorySegment{start, aligned_request, tag};
    memory_segments.insert(it, segment);

    // Update total memory required
    // Note: This can be done accesing the last element in memory_segments
    //       upon every reserve, and keeping the maximum used memory
    total_memory_required = std::max(total_memory_required,
      max_available_memory - memory_segments.back().size);

    return start;
  }

  /**
   * @brief Frees the memory segment occupied by the tag.
   */
  void free(int tag) {
    if (logger::ll.verbosityLevel >= 4) {
      debug_cout << "MemoryManager: Requested to free tag " << tag << std::endl;
    }

    auto it = std::find_if(memory_segments.begin(), memory_segments.end(),
      [&tag] (const MemorySegment& segment) {
        return segment.tag == tag;
    });

    if (it == memory_segments.end()) {
      throw StrException("Free: Requested tag could not be found ("
        + std::to_string(tag) + ")");
    }

    // Free found tag
    it->tag = -1;

    // Check if previous segment is free, in which case, join
    if (it != memory_segments.begin()) {
      auto previous_it = std::prev(it);
      if (previous_it->tag == -1) {
        previous_it->size += it->size;
        // Remove current element, and point to previous one
        it = memory_segments.erase(it);
        it--;
      }
    }

    // Check if next segment is free, in which case, join
    if (std::next(it) != memory_segments.end()) {
      auto next_it = std::next(it);
      if (next_it->tag == -1) {
        it->size += next_it->size;
        // Remove next tag
        memory_segments.erase(next_it);
      }
    }
  }

  /**
   * @brief Frees all memory segments, effectively resetting the
   *        available space.
   */
  void free_all() {
    memory_segments = std::list<MemorySegment>{{0, max_available_memory, -1}};
  }

  /**
   * @brief Prints the current state of the memory segments.
   */
  template<typename T = std::tuple<>, typename R = std::tuple<>>
  void print(
    const std::array<std::string, std::tuple_size<T>::value>& sequence_names = {},
    const std::array<std::string, std::tuple_size<R>::value>& argument_names = {},
    const int step = -1
  ) {
    if (step!=-1) {
      info_cout << "Sequence step " << step << " \""
        << sequence_names[step] << "\" memory segments (MiB):" << std::endl;
    } else {
      info_cout << "Memory segments (MiB):" << std::endl;
    }

    if (argument_names.empty()) {
      for (auto& segment : memory_segments) {
        std::string name = segment.tag==-1 ? "unused" : std::to_string(segment.tag);
        info_cout << name << " (" << ((float) segment.size) / (1024 * 1024) << "), ";
      }
      info_cout << std::endl;
    } else {
      for (auto& segment : memory_segments) {
        std::string name = segment.tag==-1 ? "unused" : argument_names[segment.tag];
        info_cout << name << " (" << ((float) segment.size) / (1024 * 1024) << "), ";
      }
      info_cout << std::endl;
    }

    info_cout << "Max memory required: "
      << (((float) total_memory_required) / (1024 * 1024)) << " MiB" << std::endl << std::endl;
  }
};
